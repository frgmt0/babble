// Native CPU inference engine for booper's Mixtral snapshots (`BABBLE_HF_RUNTIME=native`).
//
// Built on first use by babble/_native/__init__.py (g++, cached per source hash,
// flags and CPU features) and driven through the plain C ABI at the bottom of
// this file via ctypes. Model geometry is a runtime parameter (checked against
// config.json at load), so one build serves any top-1 Mixtral whose dimensions
// are multiples of 16 -- including longer-context variants.
//
// W8A32: int8 weights with a per-output-channel fp32 scale, dequantized in
// registers, fp32 activations/accumulation. Every weight matrix is repacked
// once at load into a 16-row panel layout:
//
//     panel[nb][k][j] = W[nb*16 + j][k]        (int8, 16 bytes per k)
//
// so one kernel serves both decode (M = number of streams, weights streamed
// once from DRAM for all streams) and prefill (M = prompt tokens, the panel
// stays in L1 while every 6-row tile of activations passes over it). The
// per-row scale is applied once in the epilogue.
//
// Threading: a small pool whose threads all execute the same forward code and
// meet at spin barriers (~5 per layer). A whole generate() -- prefill, every
// decode step and the sampling -- is a single parallel region. The engine is
// NOT thread-safe: callers serialize calls (NativeGenerator holds a lock).
//
// KV layout per (layer, head): a prompt ("prefix") region of Pcap positions
// shared by every stream, and per-stream regions of Scap positions for the
// generated tokens. K is blocked by 16 positions -- [pos/16][HD][16] -- so
// q.K^T is the same 16-wide register tile as the GEMMs; V is row-major
// [pos][HD]. The prompt region can be exported after a prefill and imported
// before the next one (prefix KV reuse across conversation turns): only the
// suffix of the new prompt is then prefilled.

#include <immintrin.h>
#include <linux/futex.h>
#include <pthread.h>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#define BABBLE_NATIVE_ABI 4

namespace {

constexpr int SMALL_M = 8;   // at or below this many rows, norms/routing are computed per-thread
constexpr int MAX_NE = 64;   // experts per layer (stack arrays)
constexpr int KBUF = 256;    // top-k up to this uses the streaming threshold; larger uses nth_element

template <class T>
T* amalloc(size_t n) {
  void* p = nullptr;
  if (posix_memalign(&p, 64, n * sizeof(T) + 64)) abort();
  memset(p, 0, n * sizeof(T) + 64);
  return static_cast<T*>(p);
}

inline size_t up16(size_t n) { return (n + 15) & ~size_t(15); }

double now_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ---------------------------------------------------------------- thread pool
long futex(std::atomic<uint32_t>* addr, int op, uint32_t val) {
  return syscall(SYS_futex, reinterpret_cast<uint32_t*>(addr), op, val, nullptr, nullptr, 0);
}

struct Pool {
  int n = 1;
  std::vector<std::thread> th;
  alignas(64) std::atomic<uint32_t> gen{0};
  alignas(64) std::atomic<int> pending{0};
  alignas(64) std::atomic<int> bar_count{0};
  alignas(64) std::atomic<uint32_t> bar_gen{0};
  alignas(64) void (*fn)(void*, int) = nullptr;
  void* arg = nullptr;
  std::atomic<bool> quit{false};

  void start(int nthreads) {
    n = std::max(1, nthreads);
    for (int t = 1; t < n; ++t) th.emplace_back([this, t] { worker(t); });
  }
  void stop() {
    quit = true;
    gen.fetch_add(1, std::memory_order_release);
    futex(&gen, FUTEX_WAKE_PRIVATE, INT_MAX);
    for (auto& t : th) t.join();
    th.clear();
  }
  void worker(int tid) {
    uint32_t seen = gen.load(std::memory_order_acquire);
    for (;;) {
      int spins = 0;
      uint32_t g;
      while ((g = gen.load(std::memory_order_acquire)) == seen) {
        if (++spins < (1 << 15))
          _mm_pause();
        else
          futex(&gen, FUTEX_WAIT_PRIVATE, seen);
      }
      seen = g;
      if (quit.load()) return;
      fn(arg, tid);
      pending.fetch_sub(1, std::memory_order_acq_rel);
    }
  }
  void run(void (*f)(void*, int), void* a) {
    fn = f;
    arg = a;
    pending.store(n - 1, std::memory_order_release);
    gen.fetch_add(1, std::memory_order_acq_rel);
    if (n > 1) futex(&gen, FUTEX_WAKE_PRIVATE, INT_MAX);
    f(a, 0);
    while (pending.load(std::memory_order_acquire) != 0) _mm_pause();
  }
  void barrier() {
    if (n == 1) return;
    uint32_t g = bar_gen.load(std::memory_order_acquire);
    if (bar_count.fetch_add(1, std::memory_order_acq_rel) == n - 1) {
      bar_count.store(0, std::memory_order_relaxed);
      bar_gen.fetch_add(1, std::memory_order_release);
    } else {
      int spins = 0;
      while (bar_gen.load(std::memory_order_acquire) == g) {
        if (++spins < (1 << 14))
          _mm_pause();
        else
          sched_yield();  // oversubscribed box: don't burn the timeslice the laggard needs
      }
    }
  }
};

// ------------------------------------------------------------------- math
inline float hsum(__m256 v) {
  __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
  lo = _mm_add_ps(lo, hi);
  lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
  lo = _mm_add_ss(lo, _mm_movehdup_ps(lo));
  return _mm_cvtss_f32(lo);
}

inline float hsum_max(__m256 v) {
  __m128 lo = _mm_max_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
  lo = _mm_max_ps(lo, _mm_movehl_ps(lo, lo));
  lo = _mm_max_ss(lo, _mm_movehdup_ps(lo));
  return _mm_cvtss_f32(lo);
}

// Cephes-style expf, ~1 ulp over the range that matters.
inline __m256 exp256(__m256 x) {
  const __m256 hi = _mm256_set1_ps(88.3762626647949f), lo = _mm256_set1_ps(-88.3762626647949f);
  x = _mm256_min_ps(_mm256_max_ps(x, lo), hi);
  __m256 fx = _mm256_fmadd_ps(x, _mm256_set1_ps(1.44269504088896341f), _mm256_set1_ps(0.5f));
  fx = _mm256_floor_ps(fx);
  x = _mm256_fnmadd_ps(fx, _mm256_set1_ps(0.693359375f), x);
  x = _mm256_fnmadd_ps(fx, _mm256_set1_ps(-2.12194440e-4f), x);
  __m256 z = _mm256_mul_ps(x, x);
  __m256 y = _mm256_set1_ps(1.9875691500E-4f);
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.3981999507E-3f));
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(8.3334519073E-3f));
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(4.1665795894E-2f));
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.6666665459E-1f));
  y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(5.0000001201E-1f));
  y = _mm256_fmadd_ps(y, z, x);
  y = _mm256_add_ps(y, _mm256_set1_ps(1.0f));
  __m256i n = _mm256_cvttps_epi32(fx);
  n = _mm256_slli_epi32(_mm256_add_epi32(n, _mm256_set1_epi32(127)), 23);
  return _mm256_mul_ps(y, _mm256_castsi256_ps(n));
}

// H is a multiple of 16.
void rmsnorm(const float* x, const float* w, float* out, int H, float eps) {
  __m256 acc = _mm256_setzero_ps();
  for (int i = 0; i < H; i += 8) {
    __m256 v = _mm256_loadu_ps(x + i);
    acc = _mm256_fmadd_ps(v, v, acc);
  }
  float var = hsum(acc) / H;
  float r = 1.0f / std::sqrt(var + eps);
  __m256 rv = _mm256_set1_ps(r);
  for (int i = 0; i < H; i += 8)
    _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_loadu_ps(w + i), _mm256_mul_ps(_mm256_loadu_ps(x + i), rv)));
}

// ------------------------------------------------------------ int8 kernels
struct Mat {  // [N/16][K][16] int8 + scale[N]
  int8_t* p = nullptr;
  float* s = nullptr;
  int N = 0, K = 0;
  void alloc(int n, int k) {
    N = n;
    K = k;
    p = amalloc<int8_t>((size_t)n * k);
    s = amalloc<float>(n);
  }
  void release() {
    free(p);
    free(s);
    p = nullptr;
    s = nullptr;
  }
  const int8_t* panel(int nb) const { return p + (size_t)nb * K * 16; }
  void put_row(int dst, const int8_t* src, float scale) {
    int8_t* base = p + (size_t)(dst / 16) * K * 16 + (dst % 16);
    for (int k = 0; k < K; ++k) base[k * 16] = src[k];
    s[dst] = scale;
  }
};

inline __m256 ld8(const int8_t* p) {
  return _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p))));
}

constexpr int PREFETCH = 512;  // bytes ahead to software-prefetch in the int8 panel stream (measured +3%)
constexpr int MC = 48;         // prefill row-block: activation rows kept L2-resident while panels stream by

// Right-hand operand sources for the 16-wide register tile.
struct I8Src {  // int8 panel [K][16]
  const int8_t* p;
  static constexpr bool kI8 = true;
  void load(int k, __m256& w0, __m256& w1) const {
    const int8_t* q = p + (size_t)k * 16;
    w0 = ld8(q);
    w1 = ld8(q + 8);
  }
  void prefetch(int k) const { _mm_prefetch(reinterpret_cast<const char*>(p + (size_t)k * 16 + PREFETCH), _MM_HINT_T0); }
};
struct F32Src {  // fp32 rows of stride ld, 16 contiguous columns
  const float* p;
  int ld;
  static constexpr bool kI8 = false;
  void load(int k, __m256& w0, __m256& w1) const {
    const float* q = p + (size_t)k * ld;
    w0 = _mm256_loadu_ps(q);
    w1 = _mm256_loadu_ps(q + 8);
  }
  void prefetch(int) const {}
};

// out[m][0..16) = sum_k xs[m][k] * src[k][0..16)   (no scale). U = k-unroll
// with independent accumulators so small-M (decode) tiles aren't FMA-latency bound.
template <int M, int U, class Src>
inline void k16(const float* const* xs, Src src, int K, float* __restrict out) {
  __m256 a[U][M][2];
#pragma GCC unroll 8
  for (int u = 0; u < U; ++u)
#pragma GCC unroll 8
    for (int m = 0; m < M; ++m) a[u][m][0] = a[u][m][1] = _mm256_setzero_ps();
  const float* x[M];
#pragma GCC unroll 8
  for (int m = 0; m < M; ++m) x[m] = xs[m];
  int k = 0;
  for (; k + U <= K; k += U) {
    if (Src::kI8 && ((k * 16) & 63) == 0) src.prefetch(k);
#pragma GCC unroll 8
    for (int u = 0; u < U; ++u) {
      __m256 w0, w1;
      src.load(k + u, w0, w1);
#pragma GCC unroll 8
      for (int m = 0; m < M; ++m) {
        __m256 xb = _mm256_broadcast_ss(x[m] + k + u);
        a[u][m][0] = _mm256_fmadd_ps(xb, w0, a[u][m][0]);
        a[u][m][1] = _mm256_fmadd_ps(xb, w1, a[u][m][1]);
      }
    }
  }
  for (; k < K; ++k) {
    __m256 w0, w1;
    src.load(k, w0, w1);
#pragma GCC unroll 8
    for (int m = 0; m < M; ++m) {
      __m256 xb = _mm256_broadcast_ss(x[m] + k);
      a[0][m][0] = _mm256_fmadd_ps(xb, w0, a[0][m][0]);
      a[0][m][1] = _mm256_fmadd_ps(xb, w1, a[0][m][1]);
    }
  }
#pragma GCC unroll 8
  for (int m = 0; m < M; ++m) {
    __m256 s0 = a[0][m][0], s1 = a[0][m][1];
#pragma GCC unroll 8
    for (int u = 1; u < U; ++u) {
      s0 = _mm256_add_ps(s0, a[u][m][0]);
      s1 = _mm256_add_ps(s1, a[u][m][1]);
    }
    _mm256_storeu_ps(out + m * 16, s0);
    _mm256_storeu_ps(out + m * 16 + 8, s1);
  }
}

// out[m*16 + j] for m in [0,M): tiles of up to 6 rows over one 16-col source.
template <class Src>
void tile_rows(const float* const* xs, int M, Src src, int K, float* out) {
  for (int m0 = 0; m0 < M; m0 += 6) {
    int mm = std::min(6, M - m0);
    float* o = out + m0 * 16;
    switch (mm) {
      case 1: k16<1, 4>(xs + m0, src, K, o); break;
      case 2: k16<2, 2>(xs + m0, src, K, o); break;
      case 3: k16<3, 2>(xs + m0, src, K, o); break;
      case 4: k16<4, 1>(xs + m0, src, K, o); break;
      case 5: k16<5, 1>(xs + m0, src, K, o); break;
      default: k16<6, 1>(xs + m0, src, K, o); break;
    }
  }
}

inline void panel_rows(const float* const* xs, int M, const int8_t* P, int K, float* out) {
  tile_rows(xs, M, I8Src{P}, K, out);
}

// y[m][nb*16 + j] (=|+=) scale * acc
inline void epilogue(const float* acc, int M, const float* scale, float* const* ys, int col, bool add) {
  __m256 s0 = _mm256_loadu_ps(scale), s1 = _mm256_loadu_ps(scale + 8);
  for (int m = 0; m < M; ++m) {
    float* y = ys[m] + col;
    __m256 v0 = _mm256_mul_ps(_mm256_loadu_ps(acc + m * 16), s0);
    __m256 v1 = _mm256_mul_ps(_mm256_loadu_ps(acc + m * 16 + 8), s1);
    if (add) {
      v0 = _mm256_add_ps(v0, _mm256_loadu_ps(y));
      v1 = _mm256_add_ps(v1, _mm256_loadu_ps(y + 8));
    }
    _mm256_storeu_ps(y, v0);
    _mm256_storeu_ps(y + 8, v1);
  }
}

// Contiguous share [lo, hi) of n equal units for thread tid.
inline void split(int n, int nt, int tid, int& lo, int& hi) {
  lo = (int)((long)n * tid / nt);
  hi = (int)((long)n * (tid + 1) / nt);
}

// ------------------------------------------------------------------ model
struct Layer {
  float* ln1 = nullptr;
  float* ln2 = nullptr;
  float* router = nullptr;  // [NE][H] fp32
  Mat qkv, o;
  std::vector<Mat> w13;  // panels interleaved: 2j = w1 rows 16j.., 2j+1 = w3 rows 16j..
  std::vector<Mat> w2;
};

struct SampleParams {
  int greedy;
  float temperature;
  int top_k;
  float top_p;
  float repetition_penalty;
  int no_repeat_ngram;
  int eos_id;
  int stop_at_eos;
  float frequency_penalty;
  float presence_penalty;
};

struct Rng {
  uint64_t s;
  uint64_t next() {
    uint64_t z = (s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
  }
  double uniform() { return (next() >> 11) * (1.0 / 9007199254740992.0); }
};

struct Stream {
  std::vector<int32_t> hist;  // prompt + generated (penalties see both, as HF does)
  std::vector<int32_t> uniq;  // distinct tokens of hist, first-seen order
  std::vector<int32_t> cnt;   // occurrences per vocab id
  Rng rng{0};
  double logprob = 0;
  int count = 0;
  bool active = true;
  void init(int V, size_t reserve) {
    cnt.assign(V, 0);
    hist.reserve(reserve);
  }
  void push(int tok) {
    hist.push_back(tok);
    if (cnt[tok]++ == 0) uniq.push_back(tok);
  }
};

struct Thread {
  float* xn = nullptr;   // SMALL_M x H private normed rows
  float* acc = nullptr;  // Mcap x 16 kernel output
  float* acc2 = nullptr;
  float* sc = nullptr;   // attention scores (6 rows x ld) / sampler scratch (>= V)
  const float** xs = nullptr;
  float** ys = nullptr;
  float** hs = nullptr;
  int* order = nullptr;
  int off[MAX_NE + 1];
  std::vector<std::pair<float, int>> sv;  // sampler survivors
  std::vector<float> p;                   // sampler probabilities
  std::vector<float> kth;                 // top-k scratch for large k
  // attention scratch (see attend()): roped queries, fp32 staging of one K/V
  // block, scores/probabilities, and the online-softmax state of up to QBMAX rows
  float* aq = nullptr;   // QBMAX x HD
  float* akf = nullptr;  // KB x HD (blocked K)
  float* avf = nullptr;  // KB x HD (row-major V)
  float* as = nullptr;   // QBMAX x KB
  float* ao = nullptr;   // QBMAX x HD
  float* am = nullptr;   // QBMAX running max
  float* al = nullptr;   // QBMAX running sum
  float* ak = nullptr;   // HD (roped key)
};

struct Engine {
  // geometry
  int H = 0, NH = 0, HD = 0, NL = 0, NE = 0, FF = 0, V = 0, QKV = 0, maxctx = 0;
  float eps = 1e-5f, attn_scale = 0.125f;

  int nt = 1;
  Pool pool;
  std::vector<Layer> L;
  float* normf = nullptr;
  Mat lm;  // tied embedding / lm_head, panel layout
  float* cosT = nullptr;
  float* sinT = nullptr;  // [maxctx][HD/2]

  // KV storage (see KV_FP32 / KV_FP16 / KV_Q16 below). K is kept in blocks of
  // 16 positions of kblk bytes each ([HD][16] elements, then 16 fp32 scales for
  // q16); V rows are vrow bytes. Regions are raw bytes, typed by the kernels.
  int kvmode = 0;
  size_t kblk = 0, vrow = 0;
  char *Kp = nullptr, *Vp = nullptr, *Ks = nullptr, *Vs = nullptr;
  int Pcap = 0, Scap = 0, Smax = 0, P = 0;

  // shared activations, Mcap rows
  int Mcap = 0;
  float *x = nullptr, *xn = nullptr, *qkv = nullptr, *att = nullptr, *hbuf = nullptr;
  int* expert = nullptr;
  float* logits = nullptr;
  int Lcap = 0;  // logits rows
  std::vector<Thread> tl;

  // per-call job
  const int32_t* tokens = nullptr;
  int M = 0;
  int start = 0;             // prefill: position of row 0 (prefix already in the cache)
  bool decode = false;
  const int* sid = nullptr;  // decode: row -> stream
  int g = 0;                 // decode: index in stream cache
  int logits_from = 0;       // rows [logits_from, M) get logits

  char* kp(int l, int h) { return Kp + ((size_t)l * NH + h) * (Pcap / 16) * kblk; }
  char* vp(int l, int h) { return Vp + ((size_t)l * NH + h) * Pcap * vrow; }
  char* ks(int l, int s, int h) { return Ks + (((size_t)l * Smax + s) * NH + h) * (Scap / 16) * kblk; }
  char* vs(int l, int s, int h) { return Vs + (((size_t)l * Smax + s) * NH + h) * Scap * vrow; }

  // decode split-K partials: [NH][chunks][rows][HD + 2] = (max, sum, unnormalized out[HD])
  float* part = nullptr;
  size_t part_cap = 0;
  int part_chunks = 0, part_rows = 0;
  std::atomic<int>* wq = nullptr;  // 2 work counters per layer (dynamic attention scheduling)
  void ensure_parts(int rows, int chunks) {
    const size_t need = (size_t)NH * chunks * rows * (HD + 2);
    part_chunks = chunks;
    part_rows = rows;
    if (need <= part_cap) return;
    free(part);
    part_cap = need;
    part = amalloc<float>(need);
  }
  float* part_at(int h, int c, int m) { return part + (((size_t)h * part_chunks + c) * part_rows + m) * (HD + 2); }

  void ensure_rows(int m) {
    if (m <= Mcap) return;
    free(x); free(xn); free(qkv); free(att); free(hbuf); free(expert);
    Mcap = m;
    x = amalloc<float>((size_t)m * H);
    xn = amalloc<float>((size_t)m * H);
    qkv = amalloc<float>((size_t)m * QKV);
    att = amalloc<float>((size_t)m * H);
    hbuf = amalloc<float>((size_t)m * FF);
    expert = amalloc<int>(m);
    for (auto& t : tl) {
      free(t.acc); free(t.acc2); free(t.xs); free(t.ys); free(t.hs); free(t.order);
      t.acc = amalloc<float>((size_t)m * 16 + 96);
      t.acc2 = amalloc<float>((size_t)m * 16 + 96);
      t.xs = amalloc<const float*>(m);
      t.ys = amalloc<float*>(m);
      t.hs = amalloc<float*>(m);
      t.order = amalloc<int>(m);
    }
  }
  void ensure_logits(int rows) {
    if (rows <= Lcap) return;
    free(logits);
    Lcap = rows;
    logits = amalloc<float>((size_t)rows * V);
  }
  // Grows (and so clears) the prompt KV region; call before importing a prefix.
  void ensure_prefix(int p) {
    p = (int)up16(p);
    if (p <= Pcap) return;
    free(Kp); free(Vp);
    Pcap = p;
    Kp = amalloc<char>((size_t)NL * NH * (p / 16) * kblk);
    Vp = amalloc<char>((size_t)NL * NH * p * vrow);
  }
  void ensure_streams(int s, int cap) {
    cap = (int)up16(cap);
    if (s <= Smax && cap <= Scap) return;
    free(Ks); free(Vs);
    Smax = std::max(s, Smax);
    Scap = std::max(cap, Scap);
    Ks = amalloc<char>((size_t)NL * Smax * NH * (Scap / 16) * kblk);
    Vs = amalloc<char>((size_t)NL * Smax * NH * Scap * vrow);
  }
  // Switch the KV storage type. Drops every KV buffer (snapshots taken in
  // another type are not importable afterwards).
  void set_kv_mode(int mode);

  // Packed prefix snapshot: per (layer, head), the up16(P)/16 K blocks, then
  // P V rows, in the engine's KV storage type.
  size_t kv_unit(int p) const { return up16(p) / 16 * kblk + (size_t)p * vrow; }
  size_t kv_bytes(int p) const { return (size_t)NL * NH * kv_unit(p); }

  // (l,h) units [lo, hi) of the snapshot <-> the prompt KV region
  void kv_export(int p, char* out, int lo, int hi) {
    const size_t unit = kv_unit(p), kreg = (size_t)(Pcap / 16) * kblk, vreg = (size_t)Pcap * vrow;
    const size_t kb = up16(p) / 16 * kblk;
    for (int u = lo; u < hi; ++u) {
      char* dst = out + (size_t)u * unit;
      memcpy(dst, Kp + u * kreg, kb);
      memcpy(dst + kb, Vp + u * vreg, (size_t)p * vrow);
    }
  }
  // Import positions [0, n) of a snapshot taken at length `stored` (n <= stored).
  void kv_import(const char* in, int stored, int n, int lo, int hi) {
    const size_t unit = kv_unit(stored), kreg = (size_t)(Pcap / 16) * kblk, vreg = (size_t)Pcap * vrow;
    const size_t kb_stored = up16(stored) / 16 * kblk;
    for (int u = lo; u < hi; ++u) {
      const char* src = in + (size_t)u * unit;
      memcpy(Kp + u * kreg, src, up16(n) / 16 * kblk);
      memcpy(Vp + u * vreg, src + kb_stored, (size_t)n * vrow);
    }
  }

  ~Engine() {
    for (auto& Ly : L) {
      free(Ly.ln1); free(Ly.ln2); free(Ly.router);
      Ly.qkv.release();
      Ly.o.release();
      for (auto& m : Ly.w13) m.release();
      for (auto& m : Ly.w2) m.release();
    }
    lm.release();
    free(normf); free(cosT); free(sinT);
    free(Kp); free(Vp); free(Ks); free(Vs); free(part);
    delete[] wq;
    free(x); free(xn); free(qkv); free(att); free(hbuf); free(expert); free(logits);
    for (auto& t : tl) {
      free(t.xn); free(t.acc); free(t.acc2); free(t.sc); free(t.xs); free(t.ys); free(t.hs); free(t.order);
      free(t.aq); free(t.akf); free(t.avf); free(t.as); free(t.ao); free(t.am); free(t.al); free(t.ak);
    }
  }
};

void rope(float* v, const float* c, const float* s, int half) {
  for (int i = 0; i < half; i += 8) {
    __m256 a = _mm256_loadu_ps(v + i), b = _mm256_loadu_ps(v + half + i);
    __m256 cc = _mm256_loadu_ps(c + i), ss = _mm256_loadu_ps(s + i);
    _mm256_storeu_ps(v + i, _mm256_fnmadd_ps(b, ss, _mm256_mul_ps(a, cc)));
    _mm256_storeu_ps(v + half + i, _mm256_fmadd_ps(a, ss, _mm256_mul_ps(b, cc)));
  }
}

// ---------------------------------------------------------------- attention
// KV storage policies (Engine::kvmode):
//   KV_FP32  K and V fp32 -- the exact reference arithmetic.
//   KV_FP16  K and V IEEE half (F16C, round-to-nearest-even).
//   KV_Q16   K int16 with one fp32 scale per (position, head) (max|k| / 32767),
//            V half. fp16's 11-bit mantissa is too coarse for K at long
//            context (q.K errors reach the logits; see the report), int16 with
//            a per-position scale is ~10x finer at the same 2 bytes/element.
//            The scale multiplies the 16 scores of a K block after the dot
//            product, so it costs one multiply per 16 positions.
// Every attention read goes through the cache, so prefill, decode and prefix
// restore all see the same stored K/V.
enum { KV_FP32 = 0, KV_FP16 = 1, KV_Q16 = 2 };
struct KVF32 { using K = float;    using V = float;    static constexpr bool kscale = false; };
struct KVF16 { using K = uint16_t; using V = uint16_t; static constexpr bool kscale = false; };
struct KVQ16 { using K = int16_t;  using V = uint16_t; static constexpr bool kscale = true; };
// (uint16_t elements are IEEE half, int16_t elements are scaled integers)

inline __m256 kv_ld8(const float* p) { return _mm256_loadu_ps(p); }
inline __m256 kv_ld8(const uint16_t* p) { return _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p))); }
inline __m256 kv_ld8(const int16_t* p) {
  return _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p))));
}
inline void kv_st8(float* p, __m256 v) { _mm256_storeu_ps(p, v); }
inline void kv_st8(uint16_t* p, __m256 v) {
  _mm_storeu_si128(reinterpret_cast<__m128i*>(p), _mm256_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
}
inline void kv_st1(float* p, float v) { *p = v; }
inline void kv_st1(uint16_t* p, float v) { *p = _cvtss_sh(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC); }

void Engine::set_kv_mode(int mode) {
  free(Kp); free(Vp); free(Ks); free(Vs);
  Kp = Vp = Ks = Vs = nullptr;
  Pcap = Scap = Smax = P = 0;
  kvmode = mode;
  const size_t e = mode == KV_FP32 ? 4 : 2;
  kblk = 16 * (size_t)HD * e + (mode == KV_Q16 ? 16 * sizeof(float) : 0);
  vrow = (size_t)HD * e;
}

constexpr int KB = 64;     // key positions per attention block (staged as fp32, L1-resident)
constexpr int QBMAX = 48;  // query rows per attention work unit
constexpr int DCH = 256;   // decode: shared-prompt positions per split-K chunk (multiple of KB)

// K for position t goes to block t/16 at [d][t%16] (+ its scale for q16); V is row-major [t][HD].
template <class P>
inline void put_k(char* base, int t, const float* k, int HD, size_t kblk) {
  char* blk = base + (size_t)(t / 16) * kblk;
  typename P::K* b = reinterpret_cast<typename P::K*>(blk) + (t % 16);
  if constexpr (P::kscale) {
    const __m256 sign = _mm256_set1_ps(-0.0f);
    __m256 mv = _mm256_setzero_ps();
    for (int d = 0; d < HD; d += 8) mv = _mm256_max_ps(mv, _mm256_andnot_ps(sign, _mm256_loadu_ps(k + d)));
    const float mx = hsum_max(mv);
    const float sc = mx > 0.0f ? mx / 32767.0f : 1.0f;
    const float inv = mx > 0.0f ? 32767.0f / mx : 0.0f;
    for (int d = 0; d < HD; ++d) {
      const float v = std::min(32767.0f, std::max(-32767.0f, k[d] * inv));
      b[d * 16] = (int16_t)std::nearbyint(v);
    }
    reinterpret_cast<float*>(blk + (size_t)HD * 16 * sizeof(int16_t))[t % 16] = sc;
  } else {
    for (int d = 0; d < HD; ++d) kv_st1(b + d * 16, k[d]);
  }
}
template <class P>
inline void put_v(char* base, int t, const float* v, int HD) {
  typename P::V* b = reinterpret_cast<typename P::V*>(base) + (size_t)t * HD;
  for (int d = 0; d < HD; d += 8) kv_st8(b + d, _mm256_loadu_ps(v + d));
}

// fp32 copies of a K / V block, made once per block and reused by every row
// tile (prefill). fp32 storage is used in place.
template <class P>
inline const char* stage_k(const char* src, int nb16, int HD, size_t kblk, float* tmp) {
  if constexpr (std::is_same<P, KVF32>::value) {
    return src;
  } else {
    for (int j = 0; j < nb16; ++j) {
      const typename P::K* k = reinterpret_cast<const typename P::K*>(src + (size_t)j * kblk);
      float* o = tmp + (size_t)j * HD * 16;
      __m256 s0 = _mm256_set1_ps(1.0f), s1 = s0;
      if constexpr (P::kscale) {
        const float* sc = reinterpret_cast<const float*>(src + (size_t)j * kblk + (size_t)HD * 16 * sizeof(int16_t));
        s0 = _mm256_loadu_ps(sc);
        s1 = _mm256_loadu_ps(sc + 8);
      }
      for (int d = 0; d < HD; ++d) {
        __m256 a = kv_ld8(k + d * 16), b = kv_ld8(k + d * 16 + 8);
        if constexpr (P::kscale) {
          a = _mm256_mul_ps(a, s0);
          b = _mm256_mul_ps(b, s1);
        }
        _mm256_store_ps(o + d * 16, a);
        _mm256_store_ps(o + d * 16 + 8, b);
      }
    }
    return reinterpret_cast<const char*>(tmp);
  }
}
inline const float* stage_v(const float* src, size_t, float*) { return src; }
inline const float* stage_v(const uint16_t* src, size_t n, float* tmp) {
  for (size_t i = 0; i < n; i += 32) {
    _mm256_store_ps(tmp + i, kv_ld8(src + i));
    _mm256_store_ps(tmp + i + 8, kv_ld8(src + i + 8));
    _mm256_store_ps(tmp + i + 16, kv_ld8(src + i + 16));
    _mm256_store_ps(tmp + i + 24, kv_ld8(src + i + 24));
  }
  return tmp;
}

// s[r*KB + j*16 + i] = scale * q_r . K[j*16 + i] for the 16-position K blocks
// j < nb16 starting at kf (kblk bytes apart). U independent accumulator sets
// over d (U | HD) keep the 1-2 row case from being FMA-latency bound.
template <int RR, int U, class P>
inline void qk_tile(const float* const* q, const char* kf, int nb16, int HD, float scale, float* s, size_t kblk) {
  const __m256 sc = _mm256_set1_ps(scale);
  for (int j = 0; j < nb16; ++j) {
    const typename P::K* kb = reinterpret_cast<const typename P::K*>(kf + (size_t)j * kblk);
    __m256 a[U][RR][2];
#pragma GCC unroll 8
    for (int u = 0; u < U; ++u)
#pragma GCC unroll 8
      for (int r = 0; r < RR; ++r) a[u][r][0] = a[u][r][1] = _mm256_setzero_ps();
    for (int d = 0; d < HD; d += U) {
#pragma GCC unroll 8
      for (int u = 0; u < U; ++u) {
        const __m256 w0 = kv_ld8(kb + (d + u) * 16), w1 = kv_ld8(kb + (d + u) * 16 + 8);
#pragma GCC unroll 8
        for (int r = 0; r < RR; ++r) {
          const __m256 xb = _mm256_broadcast_ss(q[r] + d + u);
          a[u][r][0] = _mm256_fmadd_ps(xb, w0, a[u][r][0]);
          a[u][r][1] = _mm256_fmadd_ps(xb, w1, a[u][r][1]);
        }
      }
    }
    __m256 sc0 = sc, sc1 = sc;
    if constexpr (P::kscale) {
      const float* ks = reinterpret_cast<const float*>(kb + (size_t)HD * 16);
      sc0 = _mm256_mul_ps(sc, _mm256_loadu_ps(ks));
      sc1 = _mm256_mul_ps(sc, _mm256_loadu_ps(ks + 8));
    }
#pragma GCC unroll 8
    for (int r = 0; r < RR; ++r) {
      __m256 s0 = a[0][r][0], s1 = a[0][r][1];
#pragma GCC unroll 8
      for (int u = 1; u < U; ++u) {
        s0 = _mm256_add_ps(s0, a[u][r][0]);
        s1 = _mm256_add_ps(s1, a[u][r][1]);
      }
      _mm256_storeu_ps(s + r * KB + j * 16, _mm256_mul_ps(s0, sc0));
      _mm256_storeu_ps(s + r * KB + j * 16 + 8, _mm256_mul_ps(s1, sc1));
    }
  }
}

// o[r][0..HD) += sum_{t < kb} p[r*KB + t] * V[t][:] (row-major V tile).
template <int RR, int U, class ST>
inline void pv_tile(const float* p, const ST* vf, int kb, int HD, float* o) {
  for (int d0 = 0; d0 < HD; d0 += 16) {
    __m256 a[U][RR][2];
#pragma GCC unroll 8
    for (int r = 0; r < RR; ++r) {
      a[0][r][0] = _mm256_loadu_ps(o + r * HD + d0);
      a[0][r][1] = _mm256_loadu_ps(o + r * HD + d0 + 8);
#pragma GCC unroll 8
      for (int u = 1; u < U; ++u) a[u][r][0] = a[u][r][1] = _mm256_setzero_ps();
    }
    int t = 0;
    for (; t + U <= kb; t += U) {
#pragma GCC unroll 8
      for (int u = 0; u < U; ++u) {
        const ST* v = vf + (size_t)(t + u) * HD + d0;
        const __m256 w0 = kv_ld8(v), w1 = kv_ld8(v + 8);
#pragma GCC unroll 8
        for (int r = 0; r < RR; ++r) {
          const __m256 pb = _mm256_broadcast_ss(p + r * KB + t + u);
          a[u][r][0] = _mm256_fmadd_ps(pb, w0, a[u][r][0]);
          a[u][r][1] = _mm256_fmadd_ps(pb, w1, a[u][r][1]);
        }
      }
    }
    for (; t < kb; ++t) {
      const ST* v = vf + (size_t)t * HD + d0;
      const __m256 w0 = kv_ld8(v), w1 = kv_ld8(v + 8);
#pragma GCC unroll 8
      for (int r = 0; r < RR; ++r) {
        const __m256 pb = _mm256_broadcast_ss(p + r * KB + t);
        a[0][r][0] = _mm256_fmadd_ps(pb, w0, a[0][r][0]);
        a[0][r][1] = _mm256_fmadd_ps(pb, w1, a[0][r][1]);
      }
    }
#pragma GCC unroll 8
    for (int r = 0; r < RR; ++r) {
      __m256 s0 = a[0][r][0], s1 = a[0][r][1];
#pragma GCC unroll 8
      for (int u = 1; u < U; ++u) {
        s0 = _mm256_add_ps(s0, a[u][r][0]);
        s1 = _mm256_add_ps(s1, a[u][r][1]);
      }
      _mm256_storeu_ps(o + r * HD + d0, s0);
      _mm256_storeu_ps(o + r * HD + d0 + 8, s1);
    }
  }
}

template <class P>
inline void qk_rows(int rr, const float* const* q, const char* kf, int nb16, int HD, float scale, float* s, size_t kblk) {
  switch (rr) {
    case 1: qk_tile<1, 4, P>(q, kf, nb16, HD, scale, s, kblk); break;
    case 2: qk_tile<2, 2, P>(q, kf, nb16, HD, scale, s, kblk); break;
    case 3: qk_tile<3, 2, P>(q, kf, nb16, HD, scale, s, kblk); break;
    case 4: qk_tile<4, 1, P>(q, kf, nb16, HD, scale, s, kblk); break;
    case 5: qk_tile<5, 1, P>(q, kf, nb16, HD, scale, s, kblk); break;
    default: qk_tile<6, 1, P>(q, kf, nb16, HD, scale, s, kblk); break;
  }
}
template <class ST>
inline void pv_rows(int rr, const float* p, const ST* vf, int kb, int HD, float* o) {
  switch (rr) {
    case 1: pv_tile<1, 4, ST>(p, vf, kb, HD, o); break;
    case 2: pv_tile<2, 2, ST>(p, vf, kb, HD, o); break;
    case 3: pv_tile<3, 2, ST>(p, vf, kb, HD, o); break;
    case 4: pv_tile<4, 1, ST>(p, vf, kb, HD, o); break;
    case 5: pv_tile<5, 1, ST>(p, vf, kb, HD, o); break;
    default: pv_tile<6, 1, ST>(p, vf, kb, HD, o); break;
  }
}

// In place over one score row: s[t] = exp(s[t] - m) for t < valid, 0 for
// t in [valid, kb). Returns the sum. kb <= KB; the row has KB slots.
inline float exp_row(float* s, int valid, int kb, float m) {
  const __m256 mv = _mm256_set1_ps(m);
  __m256 sum = _mm256_setzero_ps();
  int t = 0;
  for (; t + 8 <= valid; t += 8) {
    const __m256 e = exp256(_mm256_sub_ps(_mm256_loadu_ps(s + t), mv));
    _mm256_storeu_ps(s + t, e);
    sum = _mm256_add_ps(sum, e);
  }
  if (t < valid) {  // masked last vector: lanes >= valid become exactly 0
    const __m256i lane = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    const __m256 keep = _mm256_castsi256_ps(_mm256_cmpgt_epi32(_mm256_set1_epi32(valid - t), lane));
    const __m256 e = _mm256_and_ps(exp256(_mm256_sub_ps(_mm256_loadu_ps(s + t), mv)), keep);
    _mm256_storeu_ps(s + t, e);
    sum = _mm256_add_ps(sum, e);
    t += 8;
  }
  for (; t < kb; t += 8) _mm256_storeu_ps(s + t, _mm256_setzero_ps());
  return hsum(sum);
}

// Online-softmax (flash) attention of R <= QBMAX query rows over key positions
// [t0, t1) of one KV region (t0 a multiple of 16). Row r sees positions
// t <= lim[r] (lim ascending; nullptr = no mask). State: T.am / T.al / T.ao,
// initialised by attn_begin. All rows of a unit consume each K/V block while
// it is hot, so a unit reads each K/V line exactly once.
inline void attn_begin(Thread& T, int R, int HD) {
  for (int r = 0; r < R; ++r) {
    T.am[r] = -INFINITY;
    T.al[r] = 0.0f;
  }
  memset(T.ao, 0, (size_t)R * HD * sizeof(float));
}

template <class P>
void attend(Thread& T, const float* const* qs, int R, const int* lim, const char* Kb, const char* Vb, int t0, int t1,
            int HD, float scale, size_t kblk) {
  float* s = T.as;
  // one row tile (decode): read the stored K/V directly; several row tiles
  // (prefill): convert each block to fp32 once and reuse it for every tile
  const bool direct = R <= 6;
  for (int b0 = t0; b0 < t1; b0 += KB) {
    if (lim && lim[R - 1] < b0) break;
    const int kb = std::min(KB, t1 - b0), nb16 = (kb + 15) / 16;
    const char* kd = Kb + (size_t)(b0 / 16) * kblk;
    const typename P::V* vd = reinterpret_cast<const typename P::V*>(Vb) + (size_t)b0 * HD;
    const char* kf = direct ? nullptr : stage_k<P>(kd, nb16, HD, kblk, T.akf);
    const float* vf = direct ? nullptr : stage_v(vd, (size_t)kb * HD, T.avf);
    for (int r0 = 0; r0 < R; r0 += 6) {
      const int rr = std::min(6, R - r0);
      if (lim && lim[r0 + rr - 1] < b0) continue;  // this row tile is fully masked here (later tiles may not be)
      float* sr = s + (size_t)r0 * KB;
      if (direct) qk_rows<P>(rr, qs + r0, kd, nb16, HD, scale, sr, kblk);
      else qk_rows<KVF32>(rr, qs + r0, kf, nb16, HD, scale, sr, (size_t)HD * 16 * sizeof(float));
      for (int r = r0; r < r0 + rr; ++r) {
        float* row = s + (size_t)r * KB;
        const int valid = lim ? std::min(kb, lim[r] - b0 + 1) : kb;
        if (valid <= 0) {
          for (int t = 0; t < kb; t += 8) _mm256_storeu_ps(row + t, _mm256_setzero_ps());
          continue;
        }
        float mx = row[0];
        {
          __m256 mv = _mm256_set1_ps(-INFINITY);
          int t = 0;
          for (; t + 8 <= valid; t += 8) mv = _mm256_max_ps(mv, _mm256_loadu_ps(row + t));
          mx = std::max(mx, hsum_max(mv));
          for (; t < valid; ++t) mx = std::max(mx, row[t]);
        }
        const float mold = T.am[r], mnew = std::max(mold, mx);
        const float sum = exp_row(row, valid, kb, mnew);
        if (mnew != mold) {
          const float corr = std::exp(mold - mnew);  // 0 on the first block (mold = -inf)
          T.al[r] *= corr;
          if (mold != -INFINITY) {
            const __m256 cv = _mm256_set1_ps(corr);
            float* o = T.ao + (size_t)r * HD;
            for (int d = 0; d < HD; d += 8) _mm256_storeu_ps(o + d, _mm256_mul_ps(_mm256_loadu_ps(o + d), cv));
          }
          T.am[r] = mnew;
        }
        T.al[r] += sum;
      }
      if (direct) pv_rows(rr, sr, vd, kb, HD, T.ao + (size_t)r0 * HD);
      else pv_rows(rr, sr, vf, kb, HD, T.ao + (size_t)r0 * HD);
    }
  }
}

// Attention for layer l over E.M rows: q/k/v in E.qkv, output in E.att.
template <class P>
void attention(Engine& E, int tid, int l) {
  Thread& T = E.tl[tid];
  const int M = E.M, nt = E.nt;
  const int H = E.H, NH = E.NH, HD = E.HD, QKV = E.QKV;
  const int half = HD / 2;
  const float ascale = E.attn_scale;
  const size_t kblk = E.kblk;
  const float* qs[QBMAX];
  int lim[QBMAX];
  if (E.decode) {
    // Flash-decoding split-K: units are (head, chunk of the shared prompt, row
    // tile) -- every stream's query scores the chunk while it is hot, so the
    // prompt K/V is read once per step for all streams and spread over all
    // threads -- plus one unit per (head, row) for that stream's generated
    // tail, which also does the RoPE + cache append. A second pass merges the
    // per-chunk (max, sum, out) partials.
    const int P0 = E.P, g = E.g, n2 = g + 1, pos = P0 + g;
    const int C = (P0 + DCH - 1) / DCH, RT = (M + QBMAX - 1) / QBMAX;
    const int npre = NH * C * RT, nunits = npre + NH * M;
    const float* cs = E.cosT + (size_t)pos * half;
    const float* sn = E.sinT + (size_t)pos * half;
    std::atomic<int>& w = E.wq[2 * l];
    for (int u; (u = w.fetch_add(1, std::memory_order_relaxed)) < nunits;) {
      int h, c, r0, R, t0, t1;
      const char *Kb, *Vb;
      if (u < npre) {
        h = u % NH;
        c = (u / NH) % C;
        r0 = (u / NH / C) * QBMAX;
        R = std::min(QBMAX, M - r0);
        Kb = E.kp(l, h);
        Vb = E.vp(l, h);
        t0 = c * DCH;
        t1 = std::min(P0, t0 + DCH);
      } else {
        const int v = u - npre;
        h = v % NH;
        c = C;
        r0 = v / NH;
        R = 1;
        const int st = E.sid[r0];
        const float* kv = E.qkv + (size_t)r0 * QKV + H + h * HD;
        memcpy(T.ak, kv, HD * sizeof(float));
        rope(T.ak, cs, sn, half);
        char* kd = E.ks(l, st, h);
        char* vd = E.vs(l, st, h);
        put_k<P>(kd, g, T.ak, HD, kblk);
        put_v<P>(vd, g, kv + H, HD);
        Kb = kd;
        Vb = vd;
        t0 = 0;
        t1 = n2;
      }
      for (int r = 0; r < R; ++r) {
        float* q = T.aq + (size_t)r * HD;
        memcpy(q, E.qkv + (size_t)(r0 + r) * QKV + h * HD, HD * sizeof(float));
        rope(q, cs, sn, half);
        qs[r] = q;
      }
      attn_begin(T, R, HD);
      attend<P>(T, qs, R, nullptr, Kb, Vb, t0, t1, HD, ascale, kblk);
      for (int r = 0; r < R; ++r) {
        float* pp = E.part_at(h, c, r0 + r);
        pp[0] = T.am[r];
        pp[1] = T.al[r];
        memcpy(pp + 2, T.ao + (size_t)r * HD, HD * sizeof(float));
      }
    }
    E.pool.barrier();
    for (int it = tid; it < M * NH; it += nt) {
      const int m = it / NH, h = it % NH;
      float mx = -INFINITY;
      for (int c = 0; c <= C; ++c) mx = std::max(mx, E.part_at(h, c, m)[0]);
      float* out = E.att + (size_t)m * H + h * HD;
      for (int d = 0; d < HD; d += 8) _mm256_storeu_ps(out + d, _mm256_setzero_ps());
      float L = 0.0f;
      for (int c = 0; c <= C; ++c) {
        const float* pp = E.part_at(h, c, m);
        if (pp[1] == 0.0f) continue;
        const float f = std::exp(pp[0] - mx);
        L += pp[1] * f;
        const __m256 fv = _mm256_set1_ps(f);
        for (int d = 0; d < HD; d += 8)
          _mm256_storeu_ps(out + d, _mm256_fmadd_ps(fv, _mm256_loadu_ps(pp + 2 + d), _mm256_loadu_ps(out + d)));
      }
      const __m256 iv = _mm256_set1_ps(1.0f / L);
      for (int d = 0; d < HD; d += 8) _mm256_storeu_ps(out + d, _mm256_mul_ps(_mm256_loadu_ps(out + d), iv));
    }
  } else {
    const int S0 = E.start;
    for (int it = tid; it < M * NH; it += nt) {
      int m = it / NH, h = it % NH, pos = S0 + m;
      float* q = E.qkv + (size_t)m * QKV + h * HD;
      rope(q, E.cosT + (size_t)pos * half, E.sinT + (size_t)pos * half, half);
      rope(q + H, E.cosT + (size_t)pos * half, E.sinT + (size_t)pos * half, half);
      put_k<P>(E.kp(l, h), pos, q + H, HD, kblk);
      put_v<P>(E.vp(l, h), pos, q + 2 * H, HD);
    }
    E.pool.barrier();
    // Causal flash attention: units are (query block of QB rows, head), the
    // most expensive (latest) blocks first, handed out dynamically. Each staged
    // K/V block of KB positions serves all QB rows.
    int QB = QBMAX;
    if ((long)M * NH < 8L * nt * QBMAX) QB = std::min(QBMAX, std::max(6, (int)((M * NH / (8 * nt) + 5) / 6 * 6)));
    const int nqb = (M + QB - 1) / QB, nunits = nqb * NH;
    std::atomic<int>& w = E.wq[2 * l + 1];
    for (int u; (u = w.fetch_add(1, std::memory_order_relaxed)) < nunits;) {
      const int qb = nqb - 1 - u / NH, h = u % NH;
      const int i0 = qb * QB, R = std::min(QB, M - i0);
      for (int r = 0; r < R; ++r) {
        qs[r] = E.qkv + (size_t)(i0 + r) * QKV + h * HD;
        lim[r] = S0 + i0 + r;
      }
      attn_begin(T, R, HD);
      attend<P>(T, qs, R, lim, E.kp(l, h), E.vp(l, h), 0, S0 + i0 + R, HD, ascale, kblk);
      for (int r = 0; r < R; ++r) {
        float* out = E.att + (size_t)(i0 + r) * H + h * HD;
        const float* o = T.ao + (size_t)r * HD;
        const __m256 iv = _mm256_set1_ps(1.0f / T.al[r]);
        for (int d = 0; d < HD; d += 8) _mm256_storeu_ps(out + d, _mm256_mul_ps(_mm256_loadu_ps(o + d), iv));
      }
    }
  }
}

// Normalized rows for all M rows: per-thread private copy when M is small
// (no barrier), shared + barrier otherwise. Returns the row base pointer.
float* normed(Engine& E, int tid, const float* w) {
  Thread& T = E.tl[tid];
  const int H = E.H;
  if (E.M <= SMALL_M) {
    for (int m = 0; m < E.M; ++m) rmsnorm(E.x + (size_t)m * H, w, T.xn + (size_t)m * H, H, E.eps);
    return T.xn;
  }
  int lo, hi;
  split(E.M, E.nt, tid, lo, hi);
  for (int m = lo; m < hi; ++m) rmsnorm(E.x + (size_t)m * H, w, E.xn + (size_t)m * H, H, E.eps);
  E.pool.barrier();
  return E.xn;
}

inline int position(const Engine& E, int m) { return E.decode ? E.P + E.g : E.start + m; }

// One forward pass over E.M rows (prompt tokens, or one token per stream).
void forward(Engine& E, int tid) {
  Thread& T = E.tl[tid];
  const int M = E.M, nt = E.nt;
  const int H = E.H, NE = E.NE, FF = E.FF, V = E.V, QKV = E.QKV;
  int lo, hi;

  if (tid == 0)
    for (int i = 0; i < 2 * E.NL; ++i) E.wq[i].store(0, std::memory_order_relaxed);
  // embedding (dequant int8 row * scale, exactly what the reference computes)
  split(M, nt, tid, lo, hi);
  for (int m = lo; m < hi; ++m) {
    int tok = E.tokens[m];
    const int8_t* base = E.lm.p + (size_t)(tok / 16) * H * 16 + (tok % 16);
    float s = E.lm.s[tok];
    float* xr = E.x + (size_t)m * H;
    for (int k = 0; k < H; ++k) xr[k] = (float)base[k * 16] * s;
  }
  E.pool.barrier();

  // y rows (=|+=) xs rows . W^T over this thread's panels [nb0, nb1), row-blocked
  auto gemm = [&](const float* const* xs, float* const* ys, int rows, const Mat& W, int nb0, int nb1, bool add) {
    for (int m0 = 0; m0 < rows; m0 += MC) {
      int mc = std::min(MC, rows - m0);
      for (int nb = nb0; nb < nb1; ++nb) {
        panel_rows(xs + m0, mc, W.panel(nb), W.K, T.acc);
        epilogue(T.acc, mc, W.s + nb * 16, ys + m0, nb * 16, add);
      }
    }
  };

  for (int l = 0; l < E.NL; ++l) {
    Layer& Ly = E.L[l];
    // ---- attention
    float* xn = normed(E, tid, Ly.ln1);
    for (int m = 0; m < M; ++m) {
      T.xs[m] = xn + (size_t)m * H;
      T.ys[m] = E.qkv + (size_t)m * QKV;
    }
    split(QKV / 16, nt, tid, lo, hi);
    gemm(T.xs, T.ys, M, Ly.qkv, lo, hi, false);
    E.pool.barrier();

    switch (E.kvmode) {
      case KV_FP16: attention<KVF16>(E, tid, l); break;
      case KV_Q16: attention<KVQ16>(E, tid, l); break;
      default: attention<KVF32>(E, tid, l); break;
    }
    E.pool.barrier();

    for (int m = 0; m < M; ++m) {
      T.xs[m] = E.att + (size_t)m * H;
      T.ys[m] = E.x + (size_t)m * H;
    }
    split(H / 16, nt, tid, lo, hi);
    gemm(T.xs, T.ys, M, Ly.o, lo, hi, true);
    E.pool.barrier();

    // ---- MoE: router softmax top-1 => weight exactly 1.0, argmax of logits
    xn = normed(E, tid, Ly.ln2);
    auto route = [&](int m) {
      const float* xr = xn + (size_t)m * H;
      int best = 0;
      float bv = -INFINITY;
      for (int e = 0; e < NE; ++e) {
        __m256 a = _mm256_setzero_ps();
        const float* w = Ly.router + (size_t)e * H;
        for (int k = 0; k < H; k += 8) a = _mm256_fmadd_ps(_mm256_loadu_ps(xr + k), _mm256_loadu_ps(w + k), a);
        float v = hsum(a);
        if (v > bv) { bv = v; best = e; }
      }
      return best;
    };
    int* ex;
    int local_ex[SMALL_M];
    if (M <= SMALL_M) {
      for (int m = 0; m < M; ++m) local_ex[m] = route(m);
      ex = local_ex;
    } else {
      split(M, nt, tid, lo, hi);
      for (int m = lo; m < hi; ++m) E.expert[m] = route(m);
      E.pool.barrier();
      ex = E.expert;
    }
    // group rows by expert (every thread builds the same grouping)
    int cnt[MAX_NE] = {0};
    for (int m = 0; m < M; ++m) cnt[ex[m]]++;
    T.off[0] = 0;
    for (int e = 0; e < NE; ++e) T.off[e + 1] = T.off[e] + cnt[e];
    int fill[MAX_NE];
    for (int e = 0; e < NE; ++e) fill[e] = T.off[e];
    for (int m = 0; m < M; ++m) T.order[fill[ex[m]]++] = m;
    for (int i = 0; i < M; ++i) {
      int m = T.order[i];
      T.xs[i] = xn + (size_t)m * H;
      T.hs[i] = E.hbuf + (size_t)m * FF;
      T.ys[i] = E.x + (size_t)m * H;
    }
    // Units (expert, 16-col block) weighted by rows routed; this thread owns the
    // units whose cost midpoint falls in its share. Returns [u0, u1) per expert.
    auto my_units = [&](int per_expert, int* u0, int* u1) {
      long total = 0;
      for (int e = 0; e < NE; ++e) total += (long)cnt[e] * per_expert;
      long cum = 0;
      for (int e = 0; e < NE; ++e) {
        u0[e] = u1[e] = 0;
        bool open = false;
        for (int j = 0; j < per_expert && cnt[e]; ++j, cum += cnt[e]) {
          long mid = cum * 2 + cnt[e];
          if ((int)(mid * nt / (2 * total)) == tid) {
            if (!open) { u0[e] = j; open = true; }
            u1[e] = j + 1;
          }
        }
      }
    };
    int u0[MAX_NE], u1[MAX_NE];

    // w1/w3 + SwiGLU
    my_units(FF / 16, u0, u1);
    for (int e = 0; e < NE; ++e) {
      if (u0[e] == u1[e]) continue;
      const Mat& W = Ly.w13[e];
      const float* const* xs = T.xs + T.off[e];
      float* const* hs = T.hs + T.off[e];
      for (int m0 = 0; m0 < cnt[e]; m0 += MC) {
        const int mc = std::min(MC, cnt[e] - m0);
        for (int j = u0[e]; j < u1[e]; ++j) {
          panel_rows(xs + m0, mc, W.panel(2 * j), H, T.acc);
          panel_rows(xs + m0, mc, W.panel(2 * j + 1), H, T.acc2);
          __m256 s1a = _mm256_loadu_ps(W.s + 32 * j), s1b = _mm256_loadu_ps(W.s + 32 * j + 8);
          __m256 s3a = _mm256_loadu_ps(W.s + 32 * j + 16), s3b = _mm256_loadu_ps(W.s + 32 * j + 24);
          for (int m = 0; m < mc; ++m) {
            for (int hf = 0; hf < 2; ++hf) {
              __m256 gv = _mm256_mul_ps(_mm256_loadu_ps(T.acc + m * 16 + hf * 8), hf ? s1b : s1a);
              __m256 uv = _mm256_mul_ps(_mm256_loadu_ps(T.acc2 + m * 16 + hf * 8), hf ? s3b : s3a);
              __m256 sig = _mm256_div_ps(_mm256_set1_ps(1.0f),
                                         _mm256_add_ps(_mm256_set1_ps(1.0f), exp256(_mm256_sub_ps(_mm256_setzero_ps(), gv))));
              _mm256_storeu_ps(hs[m0 + m] + j * 16 + hf * 8, _mm256_mul_ps(_mm256_mul_ps(gv, sig), uv));
            }
          }
        }
      }
    }
    E.pool.barrier();
    my_units(H / 16, u0, u1);
    for (int e = 0; e < NE; ++e) {
      if (u0[e] == u1[e]) continue;
      gemm((const float* const*)(T.hs + T.off[e]), T.ys + T.off[e], cnt[e], Ly.w2[e], u0[e], u1[e], true);
    }
    E.pool.barrier();
  }

  // ---- final norm + tied lm_head for rows [logits_from, M)
  const int R = M - E.logits_from;
  if (R <= 0) return;
  const float* xnf;
  if (R <= SMALL_M) {
    for (int r = 0; r < R; ++r) rmsnorm(E.x + (size_t)(E.logits_from + r) * H, E.normf, T.xn + (size_t)r * H, H, E.eps);
    xnf = T.xn;
  } else {
    split(R, nt, tid, lo, hi);
    for (int r = lo; r < hi; ++r) rmsnorm(E.x + (size_t)(E.logits_from + r) * H, E.normf, E.xn + (size_t)r * H, H, E.eps);
    E.pool.barrier();
    xnf = E.xn;
  }
  for (int r = 0; r < R; ++r) {
    T.xs[r] = xnf + (size_t)r * H;
    T.ys[r] = E.logits + (size_t)r * V;
  }
  split(V / 16, nt, tid, lo, hi);
  gemm(T.xs, T.ys, R, E.lm, lo, hi, false);
  E.pool.barrier();
}

// ------------------------------------------------------------------ sampling
// Mirrors hfserve: RepetitionPenalty -> NoRepeatNGram (HF built-ins) ->
// frequency/presence (hfserve's tracker, when enabled) -> Temperature -> TopK
// (ties at the k-th value kept, like HF's `scores < kth`) -> TopP, then a
// multinomial draw from the result. The survivors are left in T.sv (ascending
// by warped score, the chosen draw range starting at *first), their
// unnormalized probabilities in T.p; returns false if nothing survives.
bool warp(const float* logits, const Stream& st, const SampleParams& sp, Thread& T, int V, size_t* first_out, double* zk_out) {
  float* buf = T.sc;
  memcpy(buf, logits, (size_t)V * sizeof(float));
  if (sp.repetition_penalty != 1.0f) {
    const float p = sp.repetition_penalty;
    for (int t : st.uniq) buf[t] = buf[t] < 0 ? buf[t] * p : buf[t] / p;
  }
  const int n = sp.no_repeat_ngram;
  const int len = (int)st.hist.size();
  if (n == 1) {
    for (int t : st.uniq) buf[t] = -INFINITY;
  } else if (n > 1 && len >= n) {
    const int32_t* hs = st.hist.data();
    const int32_t* tail = hs + len - (n - 1);
    for (int i = 0; i + n <= len; ++i) {
      bool eq = true;
      for (int j = 0; j < n - 1 && eq; ++j) eq = hs[i + j] == tail[j];
      if (eq) buf[hs[i + n - 1]] = -INFINITY;
    }
  }
  if (sp.frequency_penalty != 0.0f || sp.presence_penalty != 0.0f) {
    for (int t : st.uniq) buf[t] -= (float)st.cnt[t] * sp.frequency_penalty + sp.presence_penalty;
  }
  const float temp = sp.temperature;
  const bool scale = temp != 1.0f;
  if (scale)
    for (int i = 0; i < V; ++i) buf[i] = buf[i] / temp;
  // top-k threshold
  float thr = -INFINITY;
  const int k = sp.top_k > 0 ? std::min(sp.top_k, V) : 0;
  if (k > 0 && k < V) {
    if (k <= KBUF) {
      float top[KBUF];
      int filled = 0, minpos = 0;
      float minv = INFINITY;
      for (int i = 0; i < V; ++i) {
        float v = buf[i];
        if (filled < k) {
          top[filled++] = v;
          if (filled == k) {
            minv = top[0]; minpos = 0;
            for (int j = 1; j < k; ++j) if (top[j] < minv) { minv = top[j]; minpos = j; }
          }
        } else if (v > minv) {
          top[minpos] = v;
          minv = top[0]; minpos = 0;
          for (int j = 1; j < k; ++j) if (top[j] < minv) { minv = top[j]; minpos = j; }
        }
      }
      thr = minv;
    } else {
      T.kth.assign(buf, buf + V);
      std::nth_element(T.kth.begin(), T.kth.begin() + (k - 1), T.kth.end(), std::greater<float>());
      thr = T.kth[k - 1];
    }
  }
  auto& sv = T.sv;
  sv.clear();
  for (int i = 0; i < V; ++i) {
    float v = buf[i];
    if (v >= thr && v != -INFINITY && !std::isnan(v)) sv.emplace_back(v, i);
  }
  if (sv.empty()) return false;
  std::sort(sv.begin(), sv.end(), [](const auto& a, const auto& b) { return a.first < b.first || (a.first == b.first && a.second > b.second); });
  const float mx = sv.back().first;
  auto& p = T.p;
  p.resize(sv.size());
  double z = 0;
  for (size_t i = 0; i < sv.size(); ++i) {
    p[i] = std::exp(sv[i].first - mx);
    z += p[i];
  }
  // top-p on the ascending order, float cumsum like torch; the top token always stays
  size_t first = 0;
  if (sp.top_p < 1.0f) {
    const float cut = 1.0f - sp.top_p, zf = (float)z;
    float cum = 0;
    for (size_t i = 0; i + 1 < sv.size(); ++i) {
      cum += p[i] / zf;
      if (cum <= cut) first = i + 1;
      else break;
    }
  }
  double zk = 0;
  for (size_t i = first; i < sv.size(); ++i) zk += p[i];
  *first_out = first;
  *zk_out = zk;
  return true;
}

int sample(const float* logits, Stream& st, const SampleParams& sp, Thread& T, int V) {
  if (sp.greedy) {
    int best = 0;
    float bv = logits[0];
    for (int i = 1; i < V; ++i)
      if (logits[i] > bv) { bv = logits[i]; best = i; }
    return best;
  }
  size_t first;
  double zk;
  if (!warp(logits, st, sp, T, V, &first, &zk)) {
    // every token banned: emit eos rather than an arbitrary id
    return sp.eos_id;
  }
  const auto& sv = T.sv;
  const auto& p = T.p;
  double u = st.rng.uniform() * zk, acc = 0;
  size_t pick = sv.size() - 1;
  for (size_t i = first; i < sv.size(); ++i) {
    acc += p[i];
    if (u < acc) { pick = i; break; }
  }
  st.logprob += (double)(sv[pick].first - sv.back().first) - std::log(zk);
  return sv[pick].second;
}

// ---------------------------------------------------------------- generate
struct GenJob {
  Engine* E;
  const int32_t* prompt;
  int T, start, ns, max_new;
  const char* kv_in;
  int kv_in_len;
  char* kv_out;
  SampleParams sp;
  std::vector<Stream> st;
  std::vector<int> active;    // stream ids still generating
  std::vector<int32_t> cur;   // last token per stream
  std::vector<int32_t> toks;  // decode input per active row (written by tid 0)
  int32_t* out_tokens;
  double t0 = 0, t_first = 0, t_prefill = 0, t_last = 0;
  int steps = 0;
};

// Job fields for decode step `step` (feeds each active stream's last token;
// its K/V land at stream-cache index step-1). Called by tid 0 between barriers.
void setup_decode(GenJob& J, int step) {
  Engine& E = *J.E;
  J.toks.resize(J.active.size());
  for (size_t r = 0; r < J.active.size(); ++r) J.toks[r] = J.cur[J.active[r]];
  E.tokens = J.toks.data();
  E.M = (int)J.active.size();
  E.decode = true;
  E.sid = J.active.data();
  E.g = step - 1;
  E.logits_from = 0;
}

void import_prefix(Engine& E, const char* in, int stored, int n, int tid) {
  if (!in || n <= 0) return;
  int lo, hi;
  split(E.NL * E.NH, E.nt, tid, lo, hi);
  E.kv_import(in, stored, n, lo, hi);
  E.pool.barrier();
}

void export_prefix(Engine& E, char* out, int p, int tid) {
  if (!out) return;
  int lo, hi;
  split(E.NL * E.NH, E.nt, tid, lo, hi);
  E.kv_export(p, out, lo, hi);
  E.pool.barrier();
}

void gen_body(void* arg, int tid) {
  GenJob& J = *static_cast<GenJob*>(arg);
  Engine& E = *J.E;
  Thread& T = E.tl[tid];
  const int V = E.V;

  import_prefix(E, J.kv_in, J.kv_in_len, J.start, tid);
  // prefill the (suffix of the) shared prompt once; only the last row needs logits
  forward(E, tid);
  if (tid == 0) J.t_prefill = now_s();

  for (int s = tid; s < J.ns; s += E.nt) {
    int tok = sample(E.logits, J.st[s], J.sp, T, V);
    J.out_tokens[(size_t)s * J.max_new] = tok;
    J.cur[s] = tok;
  }
  E.pool.barrier();
  if (tid == 0) {
    J.t_first = J.t_last = now_s();
    J.active.clear();
    for (int s = 0; s < J.ns; ++s) {
      Stream& S = J.st[s];
      int tok = J.cur[s];
      S.push(tok);
      S.count = 1;
      if (J.sp.stop_at_eos && tok == J.sp.eos_id) S.active = false;
      else J.active.push_back(s);
    }
    E.P = J.T;
    J.steps = 1;
    setup_decode(J, 1);
  }
  E.pool.barrier();

  for (int step = 1; step < J.max_new; ++step) {
    if (J.active.empty()) break;  // same value on every thread (set before the barrier)
    const int B = (int)J.active.size();
    forward(E, tid);
    for (int r = tid; r < B; r += E.nt) {
      int s = J.active[r];
      int tok = sample(E.logits + (size_t)r * V, J.st[s], J.sp, T, V);
      J.out_tokens[(size_t)s * J.max_new + step] = tok;
      J.cur[s] = tok;
    }
    E.pool.barrier();
    if (tid == 0) {
      J.t_last = now_s();
      // row compaction: a stream that emitted eos leaves the batch
      std::vector<int> still;
      still.reserve(J.active.size());
      for (int s : J.active) {
        Stream& S = J.st[s];
        S.push(J.cur[s]);
        S.count++;
        if (J.sp.stop_at_eos && J.cur[s] == J.sp.eos_id) S.active = false;
        else still.push_back(s);
      }
      J.active.swap(still);
      J.steps = step + 1;
      setup_decode(J, step + 1);  // next step feeds this token at stream-cache index `step`
    }
    E.pool.barrier();
  }
  // the prompt KV region is untouched by decode: export it for the next turn
  export_prefix(E, J.kv_out, J.T, tid);
}

struct FullJob {
  Engine* E;
  const int32_t* ids;
  int T, prefill;
  float* out;
  const char* kv_in;
  int kv_in_len, start;
  char* kv_out;
};

// Prefill `prefill` tokens, then decode the rest one at a time (single stream),
// collecting logits at every position. Exercises the decode path for parity.
void incr_body(void* arg, int tid) {
  FullJob& J = *static_cast<FullJob*>(arg);
  Engine& E = *J.E;
  const int V = E.V;
  static const int zero = 0;
  forward(E, tid);  // prefill, set up by the caller
  for (int i = J.prefill; i < J.T; ++i) {
    if (tid == 0) {
      memcpy(J.out + (size_t)(i == J.prefill ? 0 : i - 1) * V, E.logits, (size_t)(i == J.prefill ? J.prefill : 1) * V * 4);
      E.P = J.prefill;
      E.tokens = J.ids + i;
      E.M = 1;
      E.start = 0;
      E.decode = true;
      E.sid = &zero;
      E.g = i - J.prefill;
      E.logits_from = 0;
    }
    E.pool.barrier();
    forward(E, tid);
    E.pool.barrier();
  }
  if (tid == 0) {
    if (J.prefill == J.T) memcpy(J.out, E.logits, (size_t)J.T * V * 4);
    else memcpy(J.out + (size_t)(J.T - 1) * V, E.logits, (size_t)V * 4);
  }
}

void full_body(void* arg, int tid) {
  FullJob& J = *static_cast<FullJob*>(arg);
  import_prefix(*J.E, J.kv_in, J.kv_in_len, J.start, tid);
  forward(*J.E, tid);
  export_prefix(*J.E, J.kv_out, J.T, tid);
}

}  // namespace

extern "C" {

int eng_abi_version() { return BABBLE_NATIVE_ABI; }

// Returns nullptr if the geometry is outside what the kernels implement.
void* eng_create(int nthreads, int hidden, int heads, int layers, int experts, int inter, int vocab, int maxctx,
                 float eps) {
  if (hidden <= 0 || heads <= 0 || hidden % heads) return nullptr;
  const int hd = hidden / heads;
  if (hidden % 16 || hd % 16 || inter % 16 || vocab % 16 || layers < 1 || experts < 1 || experts > MAX_NE ||
      maxctx < 16 || vocab < 2)
    return nullptr;
  Engine* E = new Engine();
  E->H = hidden;
  E->NH = heads;
  E->HD = hd;
  E->NL = layers;
  E->NE = experts;
  E->FF = inter;
  E->V = vocab;
  E->QKV = 3 * hidden;
  E->maxctx = maxctx;
  E->eps = eps;
  E->attn_scale = 1.0f / std::sqrt((float)hd);
  E->nt = std::max(1, nthreads);
  E->L.resize(layers);
  for (auto& Ly : E->L) {
    Ly.ln1 = amalloc<float>(hidden);
    Ly.ln2 = amalloc<float>(hidden);
    Ly.router = amalloc<float>((size_t)experts * hidden);
    Ly.qkv.alloc(E->QKV, hidden);
    Ly.o.alloc(hidden, hidden);
    Ly.w13.resize(experts);
    Ly.w2.resize(experts);
    for (int e = 0; e < experts; ++e) {
      Ly.w13[e].alloc(2 * inter, hidden);
      Ly.w2[e].alloc(hidden, inter);
    }
  }
  E->normf = amalloc<float>(hidden);
  E->lm.alloc(vocab, hidden);
  E->tl.resize(E->nt);
  for (auto& t : E->tl) {
    t.xn = amalloc<float>((size_t)SMALL_M * hidden);
    t.sc = amalloc<float>((size_t)vocab + 64);
    t.aq = amalloc<float>((size_t)QBMAX * hd);
    t.akf = amalloc<float>((size_t)KB * hd);
    t.avf = amalloc<float>((size_t)KB * hd);
    t.as = amalloc<float>((size_t)QBMAX * KB);
    t.ao = amalloc<float>((size_t)QBMAX * hd);
    t.am = amalloc<float>(QBMAX);
    t.al = amalloc<float>(QBMAX);
    t.ak = amalloc<float>(hd);
  }
  E->set_kv_mode(KV_FP32);
  E->wq = new std::atomic<int>[2 * layers];
  for (int i = 0; i < 2 * layers; ++i) E->wq[i].store(0);
  E->ensure_rows(64);
  E->pool.start(E->nt);
  return E;
}

void eng_destroy(void* p) {
  Engine* E = static_cast<Engine*>(p);
  E->pool.stop();
  delete E;
}

// kind: 0=q 1=k 2=v 3=o 4=w1 5=w2 6=w3 7=embed/lm_head
int eng_set_matrix(void* p, int kind, int layer, int expert, const int8_t* w, const float* scale, int rows, int cols) {
  Engine* E = static_cast<Engine*>(p);
  const int H = E->H, FF = E->FF;
  if (kind != 7 && (layer < 0 || layer >= E->NL)) return -3;
  if ((kind == 4 || kind == 5 || kind == 6) && (expert < 0 || expert >= E->NE)) return -3;
  auto put = [&](Mat& M, auto dst_of) {
    for (int r = 0; r < rows; ++r) M.put_row(dst_of(r), w + (size_t)r * cols, scale[r]);
  };
  Layer* Ly = kind != 7 ? &E->L[layer] : nullptr;
  switch (kind) {
    case 0: case 1: case 2:
      if (rows != H || cols != H) return -1;
      put(Ly->qkv, [&](int r) { return kind * H + r; });
      return 0;
    case 3:
      if (rows != H || cols != H) return -1;
      put(Ly->o, [](int r) { return r; });
      return 0;
    case 4: case 6:
      if (rows != FF || cols != H) return -1;
      put(Ly->w13[expert], [&](int r) { return (r / 16) * 32 + (kind == 6 ? 16 : 0) + r % 16; });
      return 0;
    case 5:
      if (rows != H || cols != FF) return -1;
      put(Ly->w2[expert], [](int r) { return r; });
      return 0;
    case 7:
      if (rows != E->V || cols != H) return -1;
      put(E->lm, [](int r) { return r; });
      return 0;
  }
  return -2;
}

// kind: 0=ln1 1=ln2 2=router[NE*H] (dequantized fp32) 3=final norm
int eng_set_vector(void* p, int kind, int layer, const float* v) {
  Engine* E = static_cast<Engine*>(p);
  if (kind != 3 && (layer < 0 || layer >= E->NL)) return -3;
  switch (kind) {
    case 0: memcpy(E->L[layer].ln1, v, (size_t)E->H * 4); return 0;
    case 1: memcpy(E->L[layer].ln2, v, (size_t)E->H * 4); return 0;
    case 2: memcpy(E->L[layer].router, v, (size_t)E->NE * E->H * 4); return 0;
    case 3: memcpy(E->normf, v, (size_t)E->H * 4); return 0;
  }
  return -2;
}

// cos/sin tables [maxctx][HD/2]
void eng_set_rope(void* p, const float* cosT, const float* sinT) {
  Engine* E = static_cast<Engine*>(p);
  const size_t n = (size_t)E->maxctx * (E->HD / 2);
  free(E->cosT);
  free(E->sinT);
  E->cosT = amalloc<float>(n);
  E->sinT = amalloc<float>(n);
  memcpy(E->cosT, cosT, n * 4);
  memcpy(E->sinT, sinT, n * 4);
}

// Bytes in a prefix snapshot of `positions` positions (depends on the KV type).
long long eng_kv_bytes(void* p, int positions) { return (long long)static_cast<Engine*>(p)->kv_bytes(positions); }

// KV storage type: 0 = fp32, 1 = fp16, 2 = q16 (K int16 + per-position scale,
// V fp16). Drops all KV state; returns the type set, or -1.
int eng_set_kv_type(void* p, int type) {
  Engine* E = static_cast<Engine*>(p);
  if (type != KV_FP32 && type != KV_FP16 && type != KV_Q16) return -1;
  E->set_kv_mode(type);
  return type;
}
int eng_kv_type(void* p) { return static_cast<Engine*>(p)->kvmode; }

// Logits for rows [start, T) of `ids` (out: [T-start][V]). Positions [0, start)
// come from the snapshot kv_in (taken at kv_in_len >= start). If kv_out is
// given, the prompt KV [0, T) is exported to it (eng_kv_floats(T) floats).
int eng_forward(void* p, const int32_t* ids, int T, int start, const void* kv_in, int kv_in_len, void* kv_out,
                float* out) {
  Engine* E = static_cast<Engine*>(p);
  if (T < 1 || T > E->maxctx || start < 0 || start >= T) return -1;
  if (start > 0 && (!kv_in || kv_in_len < start)) return -1;
  const int M = T - start;
  E->ensure_rows(M);
  E->ensure_prefix(T);
  E->ensure_logits(M);
  FullJob J{E, ids, T, T, out, static_cast<const char*>(kv_in), kv_in_len, start, static_cast<char*>(kv_out)};
  E->tokens = ids + start;
  E->M = M;
  E->start = start;
  E->decode = false;
  E->logits_from = 0;
  E->pool.run(full_body, &J);
  memcpy(out, E->logits, (size_t)M * E->V * 4);
  return 0;
}

// Prefill `prefill` tokens then single-token decode for the rest. out: [T][V]
int eng_forward_incremental(void* p, const int32_t* ids, int T, int prefill, float* out) {
  Engine* E = static_cast<Engine*>(p);
  if (T < 1 || T > E->maxctx || prefill < 1 || prefill > T) return -1;
  E->ensure_rows(prefill);
  E->ensure_prefix(prefill);
  E->ensure_streams(1, T - prefill + 1);
  E->ensure_parts(1, (prefill + DCH - 1) / DCH + 1);
  E->ensure_logits(prefill);
  FullJob J{E, ids, T, prefill, out, nullptr, 0, 0, nullptr};
  E->tokens = ids;
  E->M = prefill;
  E->start = 0;
  E->decode = false;
  E->logits_from = 0;
  E->pool.run(incr_body, &J);
  return 0;
}

// Generate `ns` streams from one shared prompt prefill.
// Prompt positions [0, start) are imported from kv_in (a snapshot of kv_in_len
// >= start positions); only [start, T) is prefilled. kv_out (optional,
// eng_kv_floats(T) floats) receives the prompt KV for the next turn.
// out_tokens [ns][max_new]; counts [ns] (tokens emitted, eos included);
// logprob [ns] (sum of the chosen tokens' post-warp log-probabilities);
// timing[0]=prefill_s [1]=first token [2]=total [3]=last token, from call entry.
// Returns the number of decode steps, or < 0 on bad arguments.
int eng_generate(void* p, const int32_t* prompt, int T, int start, const void* kv_in, int kv_in_len, void* kv_out,
                 int ns, int max_new, const SampleParams* sp, uint64_t seed, int32_t* out_tokens, int32_t* counts,
                 double* logprob, double* timing) {
  Engine* E = static_cast<Engine*>(p);
  if (T < 1 || ns < 1 || max_new < 1 || T + max_new > E->maxctx) return -1;
  if (start < 0 || start >= T || (start > 0 && (!kv_in || kv_in_len < start))) return -1;
  for (int i = 0; i < T; ++i)
    if (prompt[i] < 0 || prompt[i] >= E->V) return -2;
  if (sp->eos_id < 0 || sp->eos_id >= E->V) return -2;
  GenJob J;
  J.t0 = now_s();
  J.E = E;
  J.prompt = prompt;
  J.T = T;
  J.start = start;
  J.kv_in = static_cast<const char*>(kv_in);
  J.kv_in_len = kv_in_len;
  J.kv_out = static_cast<char*>(kv_out);
  J.ns = ns;
  J.max_new = max_new;
  J.sp = *sp;
  J.out_tokens = out_tokens;
  J.st.resize(ns);
  J.cur.assign(ns, 0);
  for (int s = 0; s < ns; ++s) {
    Stream& S = J.st[s];
    S.init(E->V, (size_t)T + max_new);
    for (int i = 0; i < T; ++i) S.push(prompt[i]);
    S.rng.s = seed * 0x100000001B3ull + (uint64_t)s * 0x9E3779B97F4A7C15ull + 1;
  }
  for (size_t i = 0; i < (size_t)ns * max_new; ++i) out_tokens[i] = sp->eos_id;
  E->ensure_rows(std::max(T - start, ns));
  E->ensure_prefix(T);
  E->ensure_streams(ns, max_new);
  E->ensure_parts(ns, (T + DCH - 1) / DCH + 1);
  E->ensure_logits(ns);
  E->tokens = prompt + start;
  E->M = T - start;
  E->start = start;
  E->decode = false;
  E->logits_from = T - start - 1;
  E->pool.run(gen_body, &J);
  double t_end = now_s();
  for (int s = 0; s < ns; ++s) {
    counts[s] = J.st[s].count;
    logprob[s] = J.st[s].logprob;
  }
  timing[0] = J.t_prefill - J.t0;
  timing[1] = J.t_first - J.t0;
  timing[2] = t_end - J.t0;
  timing[3] = J.t_last - J.t0;
  return J.steps;
}

// Sampler probe for tests: the post-warp distribution the sampler draws from,
// given `logits` [V] and the sequence so far `hist` [L]. out [V] gets the
// probabilities (0 outside the survivors). Returns the number of survivors.
int eng_warp_probs(const float* logits, int V, const int32_t* hist, int L, const SampleParams* sp, float* out) {
  Thread T;
  T.sc = amalloc<float>((size_t)V + 64);
  Stream st;
  st.init(V, L);
  for (int i = 0; i < L; ++i) st.push(hist[i]);
  size_t first;
  double zk;
  SampleParams q = *sp;
  q.greedy = 0;
  for (int i = 0; i < V; ++i) out[i] = 0.0f;
  int kept = 0;
  if (warp(logits, st, q, T, V, &first, &zk)) {
    for (size_t i = first; i < T.sv.size(); ++i) out[T.sv[i].second] = (float)(T.p[i] / zk);
    kept = (int)(T.sv.size() - first);
  }
  free(T.sc);
  return kept;
}

}  // extern "C"
