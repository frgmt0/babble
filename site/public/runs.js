// booper · runs — read-only dashboard over GET /api/runs and GET /api/runs/<name>.
// Every number on the page comes from a record the trainer (sft/sft_longform.py Reporter) POSTed.
"use strict";

const $ = (s) => document.querySelector(s);
const REFRESH_MS = 30000;
const CMP_COLORS = ["var(--s1)", "var(--s2)", "var(--s3)", "var(--s4)", "var(--s5)"];
// clip_grad_norm_(model.parameters(), 1.0) in sft_longform.py's optimizer step.
const CLIP_NORM = 1.0;
// source name in counts/source_val -> its mix / repeat CLI arg (build_examples in sft_longform.py).
const MIX_ARG = {
  tinystories: ["mix_story"], writingprompts: ["mix_wp"], no_robots: ["mix_norobots", "repeat_norobots"],
  smoltalk: ["mix_smoltalk"], discord: ["mix_discord"], ultrachat: ["mix_ultrachat"],
  dolly: ["mix_dolly", "repeat_dolly"], oasst: ["mix_oasst", "repeat_oasst"], smol_short: ["mix_smol_short"],
  nq: ["mix_nq"], arith: ["mix_arith"], persona: ["mix_persona", "repeat_persona"], wiki: ["mix_wiki"],
  discord_vibe: ["mix_discord_vibe"], booper_local: ["mix_local", "repeat_local"],
};

const state = {
  index: [], run: null, cache: new Map(), // name -> {updated, recs, model}
  cmp: [], fetchedAt: null, timer: null, inflight: false,
};

// ---------- formatting ----------
const isNum = (x) => typeof x === "number" && isFinite(x);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
function fmt(x, d = 3) {
  if (!isNum(x)) return "—";
  if (x !== 0 && Math.abs(x) < 1e-3) return x.toExponential(2);
  if (Math.abs(x) >= 1000) return Math.round(x).toLocaleString();
  return x.toFixed(d);
}
function compact(x) {
  if (!isNum(x)) return "—";
  const a = Math.abs(x);
  if (a >= 1e9) return (x / 1e9).toFixed(2) + "B";
  if (a >= 1e6) return (x / 1e6).toFixed(a >= 1e8 ? 0 : 1) + "M";
  if (a >= 1e4) return (x / 1e3).toFixed(1) + "K";
  return Math.round(x).toLocaleString();
}
const signed = (x, d = 3) => (isNum(x) ? (x > 0 ? "+" : x < 0 ? "−" : "±") + Math.abs(x).toFixed(d) : "—");
function dur(s) {
  if (!isNum(s)) return "—";
  s = Math.max(0, s);
  if (s < 90) return `${Math.round(s)}s`;
  if (s < 3600) return `${Math.round(s / 60)}m`;
  if (s < 86400 * 2) return `${Math.floor(s / 3600)}h ${String(Math.round((s % 3600) / 60)).padStart(2, "0")}m`;
  return `${(s / 86400).toFixed(1)}d`;
}
const ago = (tSec) => (isNum(tSec) ? dur(Date.now() / 1000 - tSec) + " ago" : "—");
function clock(tSec, withDay = true) {
  if (!isNum(tSec)) return "—";
  const d = new Date(tSec * 1000);
  const opts = { hour: "2-digit", minute: "2-digit" };
  if (withDay) Object.assign(opts, { weekday: "short", month: "short", day: "numeric" });
  return d.toLocaleString(undefined, opts);
}
function promptText(p) {
  if (typeof p === "string") return p;
  if (Array.isArray(p)) return p.map((t) => (typeof t === "string" ? t : t && typeof t === "object" ? `${t.role ? t.role + ": " : ""}${t.content ?? JSON.stringify(t)}` : String(t))).join("\n");
  return JSON.stringify(p);
}
function quantile(arr, q) {
  if (!arr.length) return NaN;
  const a = [...arr].sort((x, y) => x - y), i = (a.length - 1) * q, lo = Math.floor(i), hi = Math.ceil(i);
  return a[lo] + (a[hi] - a[lo]) * (i - lo);
}

// ---------- record model ----------
// A run's record list can hold several attempts: a start record with resumed_step 0 (or absent)
// begins a new attempt from scratch; a start with resumed_step R > 0 continues the attempt from
// checkpoint R, so any rows past R reported before the crash were rolled back and are dropped.
function buildModel(recs) {
  const attempts = [];
  let cur = null;
  for (const r of recs) {
    if (r.event === "start" && !(r.resumed_step > 0)) { cur = { recs: [] }; attempts.push(cur); }
    if (!cur) { cur = { recs: [] }; attempts.push(cur); }
    cur.recs.push(r);
  }
  const prior = attempts.slice(0, -1).map((a) => summarizeAttempt(a.recs));
  const m = analyzeAttempt(attempts.length ? attempts[attempts.length - 1].recs : []);
  m.prior = prior;
  m.priorTrain = attempts.slice(0, -1).map((a) => analyzeAttempt(a.recs).train);
  m.total = recs.length;
  return m;
}
function summarizeAttempt(recs) {
  const start = recs.find((r) => r.event === "start");
  const steps = recs.map((r) => r.step).filter(isNum);
  return { t: start?.t ?? recs[0]?.t, steps_total: start?.steps_total, maxStep: steps.length ? Math.max(...steps) : null, n: recs.length, lastT: recs[recs.length - 1]?.t };
}
function analyzeAttempt(recs) {
  let train = [], evals = [];
  const events = [], resumes = [], starts = [];
  let done = null;
  for (const r of recs) {
    if (r.event === "start") {
      starts.push(r);
      events.push(r);
      const R = r.resumed_step;
      if (R > 0) {
        train = train.filter((x) => x.step <= R);
        evals = evals.filter((x) => x.step <= R);
        resumes.push({ step: R, t: r.t });
      }
      continue;
    }
    if (r.event && r.event !== "done") { events.push(r); continue; }
    if ("val" in r) evals.push(r);
    else if ("loss" in r) train.push(r);
    if (r.event === "done") { done = r; events.push(r); }
  }
  train.sort((a, b) => a.step - b.step);
  evals.sort((a, b) => a.step - b.step);
  const start = starts[starts.length - 1] || null;
  const firstStart = starts[0] || null;
  const args = start?.args || {};
  const lastTrain = train[train.length - 1] || null;
  const steps_total = lastTrain?.steps_total ?? start?.steps_total ?? null;
  const lastRec = recs[recs.length - 1] || null;
  // pauses: pause events until a matching resume (or open-ended)
  const pauses = [];
  for (const e of events) {
    if (e.event === "pause") pauses.push({ t0: e.t, t1: null, step: e.step, reason: e.reason });
    if (e.event === "resume" && pauses.length && pauses[pauses.length - 1].t1 == null) pauses[pauses.length - 1].t1 = e.t;
  }
  return { recs, train, evals, events, resumes, starts, start, firstStart, done, args, lastTrain, lastRec, steps_total, pauses, counts: start?.counts || null };
}

// Typical record interval from the data: gaps between consecutive records, ignoring gaps that end
// at a (re)start or span a battery pause. Stalled = silent for > max(3 × median, 1.5 × p95).
function stallThreshold(m) {
  const gaps = [];
  for (let i = 1; i < m.recs.length; i++) {
    const a = m.recs[i - 1], b = m.recs[i];
    if (!isNum(a.t) || !isNum(b.t)) continue;
    if (b.event === "start" || a.event === "pause" || b.event === "resume") continue;
    gaps.push(b.t - a.t);
  }
  if (gaps.length < 3) return { thr: null, median: null };
  const med = quantile(gaps, 0.5), p95 = quantile(gaps, 0.95);
  return { thr: Math.max(3 * med, 1.5 * p95), median: med };
}

function runStatus(m) {
  const last = m.lastRec;
  if (m.done) return m.done.gated ? { cls: "gated", label: "done · export gated" } : { cls: "done", label: "done" };
  if (!last) return { cls: "starting", label: "no records" };
  const lastEv = m.events[m.events.length - 1];
  if (lastEv && lastEv.event === "pause" && last === lastEv) return { cls: "paused", label: `paused (${lastEv.reason || "unknown"})` };
  const { thr } = stallThreshold(m);
  const silent = Date.now() / 1000 - last.t;
  if (thr && silent > thr) return { cls: "stalled", label: "stalled", detail: `no record for ${dur(silent)} (threshold ${dur(thr)} from this run's record spacing)` };
  if (!m.train.length) return { cls: "starting", label: "starting" };
  return { cls: "running", label: "running" };
}

// Best checkpoint the trainer would have kept (consider_candidate): starts at the step-0 baseline
// when the gate is on, then any later eval with source_gate true and a lower val. The done record
// carries the trainer's own best_step / best_val and wins when present.
function bestOf(m) {
  if (m.done && isNum(m.done.best_step) && m.done.best_step >= 0) return { step: m.done.best_step, val: m.done.best_val, how: "trainer" };
  const lim = m.args.max_source_val_regression;
  if (!isNum(lim)) {
    if (!m.evals.length) return null;
    const b = m.evals.reduce((a, e) => (e.val < a.val ? e : a));
    return { step: b.step, val: b.val, how: "lowest" };
  }
  let best = null;
  if (lim >= 0) { const e0 = m.evals.find((e) => e.step === 0); if (e0) best = { step: 0, val: e0.val }; }
  for (const e of m.evals) {
    if (e.step > 0 && e.source_gate === true && isNum(e.val) && (!best || e.val < best.val)) best = { step: e.step, val: e.val };
  }
  return best ? { ...best, how: "gate" } : null;
}

function smooth(pts, mode) {
  if (mode === "off" || !pts.length) return [];
  if (mode.startsWith("ema")) {
    const a = mode === "ema97" ? 0.97 : 0.9;
    let v = 0, w = 0;
    return pts.map(([x, y]) => { v = a * v + (1 - a) * y; w = a * w + (1 - a); return [x, v / w]; });
  }
  const n = 10, out = [];
  let sum = 0;
  for (let i = 0; i < pts.length; i++) {
    sum += pts[i][1];
    if (i >= n) sum -= pts[i - n][1];
    out.push([pts[i][0], sum / Math.min(i + 1, n)]);
  }
  return out;
}

// ---------- chart primitives (hand-rolled SVG: no dependency, ~1k points is trivial) ----------
function niceTicks(a, b, n = 5) {
  if (!(b > a)) { const d = Math.abs(a) * 0.1 || 1; a -= d; b += d; }
  const raw = (b - a) / n, mag = 10 ** Math.floor(Math.log10(raw)), e = raw / mag;
  const step = (e >= 7.5 ? 10 : e >= 3.5 ? 5 : e >= 1.5 ? 2 : 1) * mag;
  const lo = Math.floor(a / step + 1e-9) * step, hi = Math.ceil(b / step - 1e-9) * step;
  const ticks = [];
  for (let v = lo; v <= hi + step / 2; v += step) ticks.push(+v.toPrecision(12));
  return { lo, hi, step, ticks };
}
function tickFmt(step) {
  if (step < 1e-3) return (v) => (v === 0 ? "0" : v.toExponential(1));
  const d = Math.max(0, -Math.floor(Math.log10(step) + 1e-9));
  return (v) => (Math.abs(v) >= 1e4 ? compact(v) : v.toFixed(d));
}
function bsearch(pts, x) { // index of nearest x in sorted pts
  let lo = 0, hi = pts.length - 1;
  while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (pts[mid][0] < x) lo = mid; else hi = mid; }
  return Math.abs(pts[lo][0] - x) <= Math.abs(pts[hi][0] - x) ? lo : hi;
}

function keyEl(color, style) {
  const k = document.createElement("i");
  k.className = "key" + (style === "faint" ? " faint" : style === "dash" ? " dash" : style === "dots" ? " dot" : style === "rule" ? " rule" : "");
  k.style.color = color;
  return k;
}
function legend(host, items) {
  const box = document.createElement("div");
  box.className = "legend";
  for (const it of items) {
    const s = document.createElement("span");
    s.append(keyEl(it.color, it.style), document.createTextNode(it.name));
    if (it.title) s.title = it.title;
    box.append(s);
  }
  host.append(box);
}

const tip = () => $("#tip");
function showTip(clientX, clientY, title, rows) {
  const t = tip();
  t.replaceChildren();
  if (title) { const d = document.createElement("div"); d.className = "t"; d.textContent = title; t.append(d); }
  for (const r of rows) {
    const d = document.createElement("div"); d.className = "row";
    if (r.color) d.append(keyEl(r.color, r.style === "dots" ? "dots" : r.style === "dash" ? "dash" : "line"));
    const b = document.createElement("b"); b.textContent = r.value;
    const s = document.createElement("span"); s.textContent = r.label;
    d.append(b, s);
    t.append(d);
  }
  t.style.display = "block";
  const w = t.offsetWidth, h = t.offsetHeight;
  let x = clientX + 14, y = clientY - h - 10;
  if (x + w > innerWidth - 8) x = clientX - w - 14;
  if (x < 8) x = 8;
  if (y < 8) y = clientY + 16;
  t.style.left = x + "px"; t.style.top = y + "px";
}
const hideTip = () => { tip().style.display = "none"; };

/**
 * series: [{name, color, pts:[[x,y]], style:'line'|'faint'|'dots'|'dash', noDomain, noLegend, noTip, dec}]
 * opts: {height, xFmt, xTip, yDec, hlines:[{y,label}], vlines:[{x,label}], bands:[{x0,x1,label}], rings:[[x,y]], xMin, xMax, yZero, title}
 */
function lineChart(host, series, opts = {}) {
  host.replaceChildren();
  series = series.map((s) => ({ ...s, pts: s.pts.filter((p) => isNum(p[0]) && isNum(p[1])).sort((a, b) => a[0] - b[0]) }));
  const shown = series.filter((s) => !s.noLegend);
  if (shown.length >= 2 || opts.hlines?.length) {
    legend(host, [...shown.map((s) => ({ name: s.name, color: s.color, style: s.style, title: s.title })),
      ...(opts.hlines || []).filter((h) => h.legend).map((h) => ({ name: h.legend, color: "var(--ink2)", style: "rule" }))]);
  }
  const all = series.filter((s) => !s.noDomain).flatMap((s) => s.pts);
  if (!all.length) { const e = document.createElement("div"); e.className = "empty"; e.textContent = opts.empty || "no data reported yet"; host.append(e); return; }
  const W = Math.max(260, host.clientWidth || 600), H = opts.height || 220;
  let x0 = opts.xMin ?? Math.min(...all.map((p) => p[0])), x1 = opts.xMax ?? Math.max(...all.map((p) => p[0]));
  if (x1 <= x0) x1 = x0 + 1;
  let ys = all.map((p) => p[1]);
  for (const h of opts.hlines || []) if (h.inDomain !== false) ys.push(h.y);
  if (opts.yZero) ys.push(0);
  let yMin = Math.min(...ys), yMax = Math.max(...ys);
  const pad = (yMax - yMin) * 0.04;
  // never pad a non-negative quantity below zero (no "-500 tok/s" ticks)
  const nt = niceTicks(yMin >= 0 && yMin - pad < 0 ? 0 : yMin - pad, yMax + pad, H < 160 ? 3 : 4);
  const yf = tickFmt(nt.step);
  const yLabels = nt.ticks.map(yf);
  const m = { l: Math.max(30, 10 + Math.max(...yLabels.map((s) => s.length)) * 6.4), r: 10, t: 8, b: 22 };
  const X = (x) => m.l + (x - x0) / (x1 - x0) * (W - m.l - m.r);
  const Y = (y) => m.t + (1 - (y - nt.lo) / (nt.hi - nt.lo)) * (H - m.t - m.b);
  const xf = opts.xFmt || ((v) => compact(v));
  const id = "c" + Math.random().toString(36).slice(2, 8);
  let g = `<defs><clipPath id="${id}"><rect x="${m.l - 10}" y="${m.t - 10}" width="${W - m.l - m.r + 20}" height="${H - m.t - m.b + 12}"/></clipPath></defs>`;
  nt.ticks.forEach((v, i) => { g += `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--grid)" stroke-width="1"/><text x="${m.l - 6}" y="${Y(v) + 4}" text-anchor="end">${esc(yLabels[i])}</text>`; });
  const nx = Math.max(2, Math.min(6, Math.floor((W - m.l - m.r) / 90)));
  const xt = opts.xTicks ? opts.xTicks(x0, x1, nx) : niceTicks(x0, x1, nx).ticks.filter((v) => v >= x0 && v <= x1);
  for (const v of xt) g += `<text x="${X(v)}" y="${H - 6}" text-anchor="middle">${esc(xf(v))}</text>`;
  g += `<line x1="${m.l}" x2="${W - m.r}" y1="${H - m.b}" y2="${H - m.b}" stroke="var(--axis)"/>`;
  for (const b of opts.bands || []) {
    const a = X(Math.max(x0, b.x0)), z = X(Math.min(x1, b.x1 ?? x1));
    if (z > a) g += `<rect x="${a}" y="${m.t}" width="${z - a}" height="${H - m.t - m.b}" fill="var(--warn)" opacity=".12"><title>${esc(b.label)}</title></rect>`;
  }
  for (const v of opts.vlines || []) {
    if (v.x < x0 || v.x > x1) continue;
    g += `<line x1="${X(v.x)}" x2="${X(v.x)}" y1="${m.t}" y2="${H - m.b}" stroke="var(--muted)" stroke-width="1"><title>${esc(v.label)}</title></line>`;
  }
  for (const h of opts.hlines || []) {
    if (h.y < nt.lo || h.y > nt.hi) continue;
    g += `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(h.y)}" y2="${Y(h.y)}" stroke="var(--ink2)" stroke-width="1.25"/>`;
    if (h.label) g += `<text class="ref" x="${m.l + 4}" y="${Y(h.y) - 4}" text-anchor="start">${esc(h.label)}</text>`;
  }
  g += `<g clip-path="url(#${id})">`;
  for (const s of series) {
    if (!s.pts.length) continue;
    if (s.style !== "dots") {
      const d = s.pts.map((p, i) => (i ? "L" : "M") + X(p[0]).toFixed(1) + " " + Y(p[1]).toFixed(1)).join("");
      const sw = s.style === "faint" ? 1 : s.width || 2;
      g += `<path d="${d}" fill="none" stroke="${s.color}" stroke-width="${sw}" stroke-opacity="${s.style === "faint" ? (s.opacity ?? 0.35) : 1}" stroke-linejoin="round" stroke-linecap="round"${s.style === "dash" ? ' stroke-dasharray="5 4"' : ""}/>`;
    }
    if (s.style === "dots" || (s.endDot && s.pts.length)) {
      const pts = s.style === "dots" ? s.pts : [s.pts[s.pts.length - 1]];
      g += pts.map((p) => `<circle cx="${X(p[0]).toFixed(1)}" cy="${Y(p[1]).toFixed(1)}" r="4" fill="${s.color}" stroke="var(--surface)" stroke-width="2"/>`).join("");
    }
  }
  for (const r of opts.rings || []) g += `<circle cx="${X(r[0])}" cy="${Y(r[1])}" r="8" fill="none" stroke="var(--ink)" stroke-width="1.5"/>`;
  g += `</g><line class="xh" x1="0" x2="0" y1="${m.t}" y2="${H - m.b}" stroke="var(--ink2)" stroke-width="1" visibility="hidden"/><g class="hov"></g>`;
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("height", H);
  svg.setAttribute("tabindex", "0");
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", opts.title || series.map((s) => s.name).join(", "));
  svg.innerHTML = g;
  host.append(svg);

  // hover: crosshair snaps to the nearest x present in any series; tooltip lists every series there.
  const tipSeries = series.filter((s) => !s.noTip && s.pts.length);
  const xsAll = [...new Set(tipSeries.flatMap((s) => s.pts.map((p) => p[0])))].sort((a, b) => a - b).map((x) => [x]);
  if (!xsAll.length) return;
  const xh = svg.querySelector(".xh"), hov = svg.querySelector(".hov");
  let idx = xsAll.length - 1;
  const show = (i, cx, cy) => {
    idx = Math.max(0, Math.min(xsAll.length - 1, i));
    const x = xsAll[idx][0];
    xh.setAttribute("x1", X(x)); xh.setAttribute("x2", X(x)); xh.setAttribute("visibility", "visible");
    const rows = [];
    let marks = "";
    const pxTol = (W - m.l - m.r) / 12;
    for (const s of tipSeries) {
      const p = s.pts[bsearch(s.pts, x)];
      if (Math.abs(X(p[0]) - X(x)) > pxTol) continue;
      rows.push({ color: s.color, style: s.style, value: (s.vfmt || ((v) => fmt(v, s.dec ?? opts.yDec ?? 3)))(p[1]), label: s.name + (p[0] !== x ? ` @ ${xf(p[0])}` : "") });
      if (p[1] >= nt.lo && p[1] <= nt.hi) marks += `<circle cx="${X(p[0])}" cy="${Y(p[1])}" r="4.5" fill="${s.color}" stroke="var(--surface)" stroke-width="2"/>`;
    }
    hov.innerHTML = marks;
    const r = svg.getBoundingClientRect();
    const scale = r.width / W;
    showTip(cx ?? r.left + X(x) * scale, cy ?? r.top + m.t * scale + 30, (opts.xTip || ((v) => `step ${Math.round(v).toLocaleString()}`))(x), rows);
  };
  const hide = () => { xh.setAttribute("visibility", "hidden"); hov.innerHTML = ""; hideTip(); };
  const onMove = (e) => {
    const r = svg.getBoundingClientRect();
    const px = (e.clientX - r.left) * (W / r.width);
    if (px < m.l - 10 || px > W - m.r + 10) return hide();
    const x = x0 + (px - m.l) / (W - m.l - m.r) * (x1 - x0);
    show(bsearch(xsAll, x), e.clientX, e.clientY);
  };
  svg.addEventListener("pointermove", onMove);
  svg.addEventListener("pointerdown", onMove);
  svg.addEventListener("pointerleave", hide);
  svg.addEventListener("blur", hide);
  svg.addEventListener("focus", () => show(idx));
  svg.addEventListener("keydown", (e) => {
    if (e.key === "ArrowLeft") { show(idx - 1); e.preventDefault(); }
    if (e.key === "ArrowRight") { show(idx + 1); e.preventDefault(); }
    if (e.key === "Escape") hide();
  });
}

function table(host, head, rows, opts = {}) {
  if (!rows.length) { host.innerHTML = `<div class="empty" style="padding:.6rem .8rem">${esc(opts.empty || "nothing reported")}</div>`; return; }
  host.innerHTML = `<table><thead><tr>${head.map((h) => `<th class="${h.l ? "l" : ""}"${h.title ? ` title="${esc(h.title)}"` : ""}>${esc(h.t ?? h)}</th>`).join("")}</tr></thead><tbody>${
    rows.map((r) => `<tr${r.cls ? ` class="${r.cls}"` : ""}>${r.cells.map((c) => (typeof c === "object" && c !== null ? `<td class="${c.cls || ""}"${c.style ? ` style="${c.style}"` : ""}${c.title ? ` title="${esc(c.title)}"` : ""}>${c.html ?? esc(c.t)}</td>` : `<td>${esc(c)}</td>`)).join("")}</tr>`).join("")}</tbody></table>`;
}

// ---------- data loading ----------
async function getJSON(url) {
  const r = await fetch(url, { cache: "no-store" });
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return r.json();
}
async function loadRunData(name, force = false) {
  const meta = state.index.find((r) => r.name === name);
  const c = state.cache.get(name);
  if (c && !force && meta && c.updated === meta.updated) return c;
  const recs = await getJSON(`/api/runs/${encodeURIComponent(name)}`);
  const entry = { updated: meta?.updated ?? null, recs, model: buildModel(Array.isArray(recs) ? recs : []) };
  state.cache.set(name, entry);
  return entry;
}

async function refresh(force = false) {
  if (state.inflight) return;
  state.inflight = true;
  $("#main").classList.add("loading");
  try {
    const idx = await getJSON("/api/runs");
    state.index = Array.isArray(idx) ? idx.sort((a, b) => b.updated - a.updated) : [];
    renderPicker();
    if (!state.index.length) { renderEmpty("no runs reported yet"); return; }
    if (!state.run || !state.index.some((r) => r.name === state.run)) {
      const want = new URLSearchParams(location.search).get("run") || decodeURIComponent(location.hash.slice(1));
      state.run = state.index.some((r) => r.name === want) ? want : state.index[0].name;
      $("#run").value = state.run;
    }
    const before = state.cache.get(state.run);
    const entry = await loadRunData(state.run, force);
    state.fetchedAt = Date.now();
    if (entry !== before || force) renderRun();
    else renderOverview(); // only relative times move
    await renderCompare(force);
  } catch (e) {
    $("#fetched").textContent = `fetch failed: ${e.message}`;
    console.warn(e);
    return;
  } finally {
    state.inflight = false;
    $("#main").classList.remove("loading");
  }
  stampFetched();
}
function stampFetched() {
  if (!state.fetchedAt) return;
  $("#fetched").textContent = `updated ${new Date(state.fetchedAt).toLocaleTimeString()}${$("#auto").checked ? "" : " · auto-refresh paused"}`;
}

function renderPicker() {
  const sel = $("#run");
  const html = state.index.map((r) => {
    const tag = r.done ? " ✓ done" : "";
    return `<option value="${esc(r.name)}">${esc(r.name)}${tag} — step ${r.step ?? "?"}/${r.steps_total ?? "?"}</option>`;
  }).join("");
  if (sel.innerHTML !== html) { sel.innerHTML = html; if (state.run) sel.value = state.run; }
}
function renderEmpty(msg) {
  $("#status").innerHTML = `<span class="empty">${esc(msg)}</span>`;
}

function setRun(name) {
  state.run = name;
  const u = new URL(location.href);
  u.searchParams.set("run", name);
  u.hash = "";
  history.replaceState(null, "", u);
  state.cmp = state.cmp.filter((n) => n !== name);
  refresh();
}

// ---------- rendering ----------
const model = () => state.cache.get(state.run)?.model;

function renderRun() {
  renderOverview();
  renderTraining();
  renderEval();
  renderGate();
  renderSamplesControls();
  renderData();
  renderConfig();
  renderTables();
}

function renderOverview() {
  const m = model();
  if (!m) return;
  const st = runStatus(m);
  const lt = m.lastTrain, best = bestOf(m);
  const lastEval = m.evals[m.evals.length - 1];
  const t0 = m.firstStart?.t ?? m.recs[0]?.t;
  const tEnd = m.done ? m.done.t : st.cls === "running" || st.cls === "starting" ? Date.now() / 1000 : m.lastRec?.t;
  const gateTxt = !m.done ? "" : m.done.gated === true ? "export gated: no post-base checkpoint passed" : isNum(m.done.best_step) ? `export from best step ${m.done.best_step}` : "";
  $("#status").innerHTML = `<span class="pill ${st.cls}" title="${esc(st.detail || "")}"><span class="dot"></span>${esc(st.label)}</span>
    <span class="small muted">last record ${esc(ago(m.lastRec?.t))}${st.detail ? " · " + esc(st.detail) : ""}</span>
    ${gateTxt ? `<span class="small ${m.done.gated === true ? "no" : "ok"}">${m.done.gated === true ? "✕" : "✓"} ${esc(gateTxt)}</span>` : ""}`;

  const step = m.done?.step ?? lt?.step ?? lastEval?.step ?? 0;
  const tot = m.steps_total;
  const pct = isNum(tot) && tot > 0 ? Math.min(100, (step / tot) * 100) : null;
  $("#progress").innerHTML = `<div class="track" role="progressbar" aria-valuemin="0" aria-valuemax="${tot ?? 0}" aria-valuenow="${step}"><div class="fill" style="width:${pct ?? 0}%"></div></div>
    <div class="lbl"><span>step <b>${step.toLocaleString()}</b> / ${isNum(tot) ? tot.toLocaleString() : "?"}${pct != null ? ` · ${pct.toFixed(1)}%` : ""}</span>
    <span>${m.done ? `finished ${esc(clock(m.done.t))}` : lt && isNum(lt.eta_s) ? `${st.cls === "running" ? "ETA" : "last reported ETA"} ${dur(lt.eta_s)} → ~${esc(clock(lt.t + lt.eta_s))} <span class="muted">(as of step ${lt.step}${st.cls === "running" ? "" : ", stale"})</span>` : ""}</span></div>`;

  const notes = [];
  if (m.prior.length) notes.push(`restarted from scratch ${m.prior.length}× — showing the latest attempt (started ${esc(clock(t0))}); earlier: ${m.prior.map((p) => `${esc(clock(p.t))} reached step ${p.maxStep ?? 0}/${p.steps_total ?? "?"}`).join("; ")}`);
  if (m.resumes.length) notes.push(`resumed from checkpoint at step ${m.resumes.map((r) => r.step.toLocaleString()).join(", ")}${m.starts.length > 1 && new Set(m.starts.map((s) => s.steps_total)).size > 1 ? ` (steps_total changed: ${[...new Set(m.starts.map((s) => s.steps_total))].join(" → ")})` : ""}`);
  if (m.pauses.length) notes.push(`${m.pauses.length} battery pause${m.pauses.length > 1 ? "s" : ""}, ${dur(m.pauses.reduce((a, p) => a + ((p.t1 ?? Date.now() / 1000) - p.t0), 0))} total`);
  if (m.total >= 5000) notes.push(`<span class="warn-note">this run hit the server's 5,000-record cap; the oldest train rows were dropped, so early loss history is incomplete.</span>`);
  $("#notes").innerHTML = notes.map((n) => `<li>${n}</li>`).join("");

  const sm = smooth(m.train.map((r) => [r.step, r.loss]), "ema90");
  const tiles = [
    ["train loss", fmt(sm.length ? sm[sm.length - 1][1] : null), lt ? `raw ${fmt(lt.loss)} · EMA 0.9` : ""],
    ["val loss", fmt(lastEval?.val), lastEval ? `step ${lastEval.step}${m.evals.length > 1 ? ` · ${signed(lastEval.val - m.evals[0].val)} vs step ${m.evals[0].step}` : ""}` : ""],
    ["best", best ? fmt(best.val) : "—", best ? (best.how === "gate" && best.step === 0 ? "step-0 baseline (nothing better passed yet)" : `step ${best.step} · ${best.how === "trainer" ? "trainer's pick" : best.how === "gate" ? "gate-passing" : "lowest val"}`) : ""],
    ["elapsed", dur(isNum(t0) && isNum(tEnd) ? tEnd - t0 : null), t0 ? `since ${clock(t0)}` : ""],
    ["tokens seen", compact(lt?.tokens), m.args.tokens ? `budget ${compact(m.args.tokens)}` : ""],
    ["learning rate", lt ? fmt(lt.lr) : "—", m.args.lr ? `peak ${fmt(m.args.lr)}` : ""],
    ["tok/s (wall)", lt ? compact(lt.tok_s) : "—", lt && isNum(lt.tok_s_active) ? `active ${compact(lt.tok_s_active)}` : ""],
    ["grad norm", fmt(lt?.grad_norm, 2), `pre-clip · clip at ${CLIP_NORM.toFixed(1)}`],
    ["memory", lt && isNum(lt.rss_gb) ? fmt(lt.rss_gb, 2) + " GB" : "—", "process RSS"],
    ["duty cycle", fmt(lt?.duty_cycle ?? m.args.duty_cycle ?? m.start?.duty_cycle, 2), lt && isNum(lt.idle_s) ? `idle ${dur(lt.idle_s)} last interval` : ""],
    ["device", esc(m.start?.device ?? "—"), m.args.base ? `base ${m.args.base}` : ""],
  ];
  $("#tiles").innerHTML = tiles.map(([k, v, d]) => `<div class="tile"><div class="k">${esc(k)}</div><div class="v" title="${esc(v)}">${v}</div><div class="d" title="${esc(d)}">${esc(d)}</div></div>`).join("");
  document.title = `${state.run} · ${st.label} · booper runs`;
}

function resumeLines(m) {
  return m.resumes.map((r) => ({ x: r.step, label: `resumed from step ${r.step} at ${clock(r.t)}` }));
}

function renderTraining() {
  const m = model();
  const mode = $("#smooth").value, fitY = $("#fitY").checked;
  const raw = m.train.map((r) => [r.step, r.loss]);
  const best = bestOf(m);
  const series = [{ name: "train", color: "var(--s1)", pts: raw, style: mode === "off" ? "line" : "faint", noDomain: fitY && mode !== "off", title: "mean loss over the micro-batches since the previous log row (all sources mixed)" }];
  if (mode !== "off") series.push({ name: `train (${$("#smooth").selectedOptions[0].text})`, color: "var(--s1)", pts: smooth(raw, mode) });
  series.push({ name: "val", color: "var(--s2)", pts: m.evals.map((e) => [e.step, e.val]), style: "dots" });
  if ($("#earlier").checked) {
    m.priorTrain.forEach((tr, i) => series.push({ name: `attempt ${i + 1} (train, smoothed)`, color: "var(--muted)", pts: smooth(tr.map((r) => [r.step, r.loss]), "ema90"), style: "faint", opacity: 0.7, noDomain: fitY }));
  }
  lineChart($("#c-loss"), series, { height: 280, vlines: resumeLines(m), rings: best ? [[best.step, best.val]] : [], title: "loss by step" });

  const gn = m.train.filter((r) => isNum(r.grad_norm)).map((r) => [r.step, r.grad_norm]);
  lineChart($("#c-gn"), [{ name: "grad norm", color: "var(--s1)", pts: gn, dec: 3 }], { hlines: [{ y: CLIP_NORM, label: `clip ${CLIP_NORM.toFixed(1)}` }], vlines: resumeLines(m), yZero: true, title: "gradient norm" });

  const lrPts = m.train.filter((r) => isNum(r.lr)).map((r) => [r.step, r.lr]);
  const lrSeries = [{ name: "reported lr", color: "var(--s1)", pts: lrPts, vfmt: (v) => v.toExponential(3) }];
  const a = m.args, T = m.steps_total;
  if (isNum(a.lr) && isNum(a.warmup) && isNum(T) && T > 0) {
    // lr_at(s) from sft_longform.py; the trainer logs lr_at(step - 1) on the row for `step`.
    const lrAt = (s) => (s < a.warmup ? a.lr * (s + 1) / a.warmup : a.lr * (0.1 + 0.9 * 0.5 * (1 + Math.cos(Math.PI * Math.min(1, (s - a.warmup) / Math.max(1, T - a.warmup))))));
    const plan = [];
    const n = 240;
    for (let i = 0; i <= n; i++) { const step = Math.max(1, Math.round((i / n) * T)); plan.push([step, lrAt(step - 1)]); }
    for (let s = 1; s <= Math.min(a.warmup + 1, T); s += Math.max(1, Math.floor(a.warmup / 20))) plan.push([s, lrAt(s - 1)]);
    lrSeries.unshift({ name: "planned (warmup + cosine)", color: "var(--s2)", pts: plan, style: "dash", vfmt: (v) => v.toExponential(3) });
    $("#lr-cap").textContent = `planned curve recomputed from args: peak ${a.lr}, warmup ${a.warmup} steps, cosine to 10% of peak at step ${T.toLocaleString()} (latest steps_total).`;
  } else $("#lr-cap").textContent = "";
  lineChart($("#c-lr"), lrSeries, { vlines: resumeLines(m), yZero: true, title: "learning rate" });

  const tps = [{ name: "wall", color: "var(--s1)", pts: m.train.map((r) => [r.step, r.tok_s]), dec: 0 }];
  if (m.train.some((r) => isNum(r.tok_s_active))) tps.unshift({ name: "active (excl. sleep/pauses)", color: "var(--s3)", pts: m.train.map((r) => [r.step, r.tok_s_active]), dec: 0 });
  lineChart($("#c-tps"), tps, { vlines: resumeLines(m), yZero: true, title: "throughput" });

  lineChart($("#c-rss"), [{ name: "RSS GB", color: "var(--s1)", pts: m.train.map((r) => [r.step, r.rss_gb]), dec: 2 }], { vlines: resumeLines(m), yZero: true, title: "memory" });

  // step vs wall-clock, including the earlier attempts' records so restarts are visible on the time axis.
  const cache = state.cache.get(state.run);
  const allRecs = cache.recs.filter((r) => isNum(r.t));
  const cur = m.recs.filter((r) => isNum(r.step) && isNum(r.t)).map((r) => [r.t, r.step]);
  const prev = allRecs.filter((r) => isNum(r.step) && !m.recs.includes(r)).map((r) => [r.t, r.step]);
  const tv = [];
  for (const r of allRecs) if (r.event === "start") tv.push({ x: r.t, label: `${r.resumed_step > 0 ? `resume from step ${r.resumed_step}` : "start / restart"} · ${clock(r.t)}` });
  const tSeries = [{ name: "latest attempt", color: "var(--s1)", pts: cur, dec: 0, vfmt: (v) => Math.round(v).toLocaleString() }];
  if (prev.length) tSeries.push({ name: "earlier attempts", color: "var(--muted)", pts: prev, style: "dots", vfmt: (v) => Math.round(v).toLocaleString() });
  const span = allRecs.length ? allRecs[allRecs.length - 1].t - allRecs[0].t : 0;
  lineChart($("#c-time"), tSeries, {
    vlines: tv, bands: m.pauses.map((p) => ({ x0: p.t0, x1: p.t1 ?? Date.now() / 1000, label: `paused (${p.reason})` })),
    xFmt: (v) => (span > 86400 * 1.5 ? new Date(v * 1000).toLocaleDateString(undefined, { month: "short", day: "numeric" }) : clock(v, false)),
    xTip: (v) => clock(v), yZero: true, title: "step over wall-clock time",
  });
}

// source_val keys: <src>, <src>_single, <src>_legacy, <src>_multiturn
function sourceNames(evals) {
  const keys = new Set();
  for (const e of evals) for (const k of Object.keys(e.source_val || {})) keys.add(k);
  const base = [...keys].filter((k) => !/_(single|legacy|multiturn)$/.test(k));
  return { keys, base };
}

function renderEval() {
  const m = model();
  const ev = m.evals.filter((e) => e.source_val);
  const { keys, base } = sourceNames(ev);
  const first = ev[0], last = ev[ev.length - 1];
  if (!ev.length) {
    $("#src-table").innerHTML = `<div class="empty" style="padding:.6rem .8rem">this run reported no per-source eval</div>`;
    $("#multiples").replaceChildren(); $("#mult-legend").replaceChildren();
    return;
  }
  const minOf = (k) => { let b = null; for (const e of ev) { const v = e.source_val?.[k]; if (isNum(v) && (!b || v < b.v)) b = { v, step: e.step }; } return b; };
  const views = ["", "_multiturn", "_single", "_legacy"].filter((suf) => base.some((b) => keys.has(b + suf)));
  const head = [{ t: "source", l: true }, { t: "val n", title: "val examples (start record counts)" }];
  for (const suf of views) {
    const nm = suf ? suf.slice(1) : "role";
    head.push({ t: `${nm} @${first.step}` }, { t: `${nm} @${last.step}` }, { t: "Δ" }, { t: "best (step)" });
  }
  const rows = base.sort().map((b) => {
    const cells = [b, m.counts?.[b]?.val ?? "—"];
    for (const suf of views) {
      const k = b + suf, v0 = first.source_val?.[k], v1 = last.source_val?.[k], bb = minOf(k);
      const d = isNum(v0) && isNum(v1) ? v1 - v0 : null;
      cells.push(fmt(v0, 4), fmt(v1, 4), { t: signed(d), cls: d == null ? "" : d < 0 ? "ok" : d > 0 ? "no" : "" }, bb ? `${fmt(bb.v, 4)} (${bb.step})` : "—");
    }
    return { cells };
  });
  table($("#src-table"), head, rows);

  const lg = $("#mult-legend");
  lg.replaceChildren();
  const anyMT = base.some((b) => keys.has(b + "_multiturn"));
  legend(lg, [{ name: "role view", color: "var(--s1)", style: "line" }, ...(anyMT ? [{ name: "multiturn view", color: "var(--s2)", style: "line" }] : [])]);
  lg.firstChild.style.margin = "0";
  const host = $("#multiples");
  host.replaceChildren();
  for (const b of base) {
    const card = document.createElement("div");
    card.className = "card";
    const v0 = first.source_val?.[b], v1 = last.source_val?.[b];
    card.innerHTML = `<h3><span>${esc(b)}</span><small>${isNum(v0) && isNum(v1) ? esc(signed(v1 - v0)) + " role" : ""}</small></h3><div class="chart"></div>`;
    host.append(card);
    const s = [{ name: "role", color: "var(--s1)", pts: ev.map((e) => [e.step, e.source_val?.[b]]), endDot: true, dec: 4 }];
    if (keys.has(b + "_multiturn")) s.push({ name: "multiturn", color: "var(--s2)", pts: ev.map((e) => [e.step, e.source_val?.[b + "_multiturn"]]), endDot: true, dec: 4 });
    s.forEach((x) => (x.noLegend = true));
    lineChart(card.querySelector(".chart"), s, { height: 130, title: `${b} val by step` });
  }
}

// Mirror of _source_gate / _source_gate_core in sft_longform.py.
function gateRule(key, args) {
  const lim = args.max_source_val_regression;
  if (!isNum(lim)) return null;
  if (lim < 0) return { kind: "off", label: "gate off" };
  const glim = isNum(args.guard_max_regression) ? args.guard_max_regression : lim;
  const mmi = args.multiturn_must_improve;
  if (key.endsWith("_guard_missing")) return { kind: "guard", src: key.slice(0, -14), limit: null, missing: true };
  if (key.endsWith("_guard")) return { kind: "guard", src: key.slice(0, -6), limit: glim };
  if (key.endsWith("_role")) return { kind: "role", src: key.slice(0, -5), limit: lim };
  if (key.endsWith("_migration")) return { kind: "migration", src: key.slice(0, -10), limit: lim };
  if (key.endsWith("_multiturn")) {
    const src = key.slice(0, -10);
    const must = mmi == null || (Array.isArray(mmi) && mmi.includes(src));
    return { kind: "multiturn", src, limit: must ? 0 : lim, strict: must };
  }
  return { kind: "other", src: key, limit: lim };
}
const passes = (d, r) => (!r || r.kind === "off" ? null : r.missing || !isNum(d) ? false : r.strict ? d < 0 : d <= r.limit);
const GROUPS = {
  role: "role format vs baseline (≤ max_source_val_regression)",
  migration: "migration: role single-turn vs legacy prompt (≤ max_source_val_regression)",
  multiturn: "multi-turn views (must improve, or ≤ limit when not in multiturn_must_improve)",
  guard: "guard sources (≤ guard_max_regression)",
  other: "other",
};

function renderGate() {
  const m = model();
  const ev = m.evals.filter((e) => e.source_regression && typeof e.source_regression === "object");
  const a = m.args;
  const headEl = $("#gate-head"), rowsEl = $("#gate-rows");
  if (!ev.length) {
    headEl.innerHTML = !isNum(a.max_source_val_regression) ? `<p class="sub small">this run has no source gate (its args carry no max_source_val_regression).</p>` : `<p class="sub small">${a.max_source_val_regression < 0 ? "source gate disabled (max_source_val_regression < 0). " : ""}${m.evals.length ? "no gated eval yet — the step-0 eval is the baseline; regressions appear from the first later eval." : "this run reported no evals."}${isNum(a.max_source_val_regression) ? ` limit ${a.max_source_val_regression}` : ""}</p>`;
    rowsEl.innerHTML = ""; $("#gate-heat").innerHTML = "";
    return;
  }
  if (!ev.some((e) => e.step === Number(headEl.dataset.step))) headEl.dataset.step = ev[ev.length - 1].step;
  const sel = ev.find((e) => e.step === Number(headEl.dataset.step));
  const verdict = sel.source_gate === true ? `<span class="verdict ok">✓ passed</span>` : sel.source_gate === false ? `<span class="verdict no">✕ failed</span>` : `<span class="verdict off">— not reported</span>`;
  headEl.innerHTML = `<div class="bar" style="margin-top:0"><label>eval step <select id="gate-step">${ev.map((e) => `<option value="${e.step}"${e === sel ? " selected" : ""}>${e.step}${e.event === "done" ? " (final)" : ""}</option>`).join("")}</select></label>
    <span>trainer's verdict at step ${sel.step}: ${verdict}</span>
    <span class="small muted">limit ${fmt(a.max_source_val_regression, 3)}${isNum(a.guard_max_regression) ? ` · guard limit ${fmt(a.guard_max_regression, 3)}` : ""}${a.guard_sources?.length ? ` · guards: ${esc(a.guard_sources.join(", "))}` : ""}${a.multiturn_must_improve?.length ? ` · multi-turn must improve: ${esc(a.multiturn_must_improve.join(", "))}` : ""}</span></div>`;
  $("#gate-step").onchange = (e) => { headEl.dataset.step = e.target.value; renderGate(); };

  const reg = sel.source_regression;
  const entries = Object.entries(reg).map(([k, d]) => ({ k, d: isNum(d) ? d : null, r: gateRule(k, a) }));
  // Scale to the limit's neighbourhood so one large improvement does not flatten every other row;
  // bars past the edge are clamped and marked (the exact value is in the number column).
  const absD = entries.map((e) => Math.abs(e.d ?? 0));
  const lims = entries.map((e) => Math.abs(e.r?.limit ?? 0));
  const M = Math.max(1e-6, Math.min(Math.max(...absD, ...lims), Math.max(4 * Math.max(...lims), 1.5 * quantile(absD, 0.75)))) * 1.08;
  const pos = (v) => `${50 + (v / M) * 50}%`;
  let html = `<div class="reg">`;
  for (const kind of ["role", "migration", "multiturn", "guard", "other"]) {
    const g = entries.filter((e) => (e.r?.kind === "off" ? "other" : e.r?.kind || "other") === kind);
    if (!g.length) continue;
    html += `<div class="grp">${esc(GROUPS[kind])}</div>`;
    for (const e of g) {
      const p = passes(e.d, e.r);
      const lim = e.r && e.r.kind !== "off" && isNum(e.r.limit) ? e.r.limit : null;
      const over = e.d != null && Math.abs(e.d) > M;
      const w = e.d == null ? 0 : Math.min(50, Math.abs(e.d) / M * 50);
      const tipTxt = `${e.k}: ${e.d == null ? "missing / infinite" : signed(e.d, 4)} nats vs baseline${lim != null ? ` · limit ${e.r.strict ? "< 0 (must improve)" : "≤ " + lim}` : ""}`;
      html += `<div class="name" title="${esc(e.k)}">${esc(e.r?.src ?? e.k)}${e.k.includes("_multiturn_guard") ? " <span class=muted>multiturn</span>" : ""}</div>
        <div class="lane" tabindex="0" title="${esc(tipTxt)}" aria-label="${esc(tipTxt)}"><span class="zero" style="left:50%"></span>${e.d == null ? "" : `<span class="b ${e.d < 0 ? "neg" : "pos"}${over ? " over" : ""}" style="${e.d < 0 ? `right:50%` : `left:50%`};width:${w}%"></span>`}${lim != null ? `<span class="lim" style="left:calc(${pos(lim)} - 1px)"></span>` : ""}</div>
        <div class="val">${esc(e.d == null ? "∞" : signed(e.d, 4))}</div>
        <div class="${p === true ? "ok" : p === false ? "no" : "off"}">${p === true ? "✓ pass" : p === false ? "✕ fail" : "— n/a"}</div>`;
    }
  }
  html += `</div>`;
  rowsEl.innerHTML = html;

  // history: keys × eval steps, diverging wash (blue = better, red = worse), ✕ on failing cells.
  const keys = [...new Set(ev.flatMap((e) => Object.keys(e.source_regression)))];
  const HM = Math.max(1e-6, ...ev.flatMap((e) => Object.values(e.source_regression).filter(isNum).map(Math.abs)));
  const cell = (d, r) => {
    if (!isNum(d)) return { t: "—" };
    const p = passes(d, r);
    const share = Math.round(Math.min(1, Math.abs(d) / HM) * 55);
    return { html: `${p === false ? "✕ " : ""}${esc(signed(d, 3))}`, style: `background:color-mix(in srgb, ${d < 0 ? "var(--div-neg)" : "var(--div-pos)"} ${share}%, var(--surface))`, title: `${p === false ? "fails" : p === true ? "passes" : ""} ${r && isNum(r.limit) ? (r.strict ? "(must be < 0)" : `(limit ${r.limit})`) : ""}` };
  };
  table($("#gate-heat"), [{ t: "key", l: true }, ...ev.map((e) => ({ t: `${e.step}${e.source_gate === false ? " ✕" : e.source_gate === true ? " ✓" : ""}` }))],
    keys.map((k) => ({ cells: [k, ...ev.map((e) => cell(e.source_regression[k], gateRule(k, a)))] })));
  $("#gate-heat").querySelector("table")?.classList.add("heat");
}

function renderSamplesControls() {
  const m = model();
  const ev = m.evals.filter((e) => Array.isArray(e.samples) && e.samples.length);
  const s = $("#s-step"), b = $("#s-base");
  if (!ev.length) { s.innerHTML = b.innerHTML = ""; $("#s-list").innerHTML = `<p class="empty">no samples reported</p>`; return; }
  const prevS = s.value, prevB = b.value;
  const opts = ev.map((e) => `<option value="${e.step}">${e.step}${e.event === "done" ? " (final)" : ""}</option>`).join("");
  s.innerHTML = opts;
  b.innerHTML = `<option value="none">nothing (single column)</option>` + opts;
  s.value = ev.some((e) => String(e.step) === prevS) ? prevS : String(ev[ev.length - 1].step);
  b.value = prevB === "none" || ev.some((e) => String(e.step) === prevB) ? prevB || String(ev[0].step) : String(ev[0].step);
  renderSamples();
}
function renderSamples() {
  const m = model();
  const ev = m.evals.filter((e) => Array.isArray(e.samples) && e.samples.length);
  const pick = (v) => [...ev].reverse().find((e) => String(e.step) === v); // final (done) wins a tie
  const cur = pick($("#s-step").value), base = $("#s-base").value === "none" ? null : pick($("#s-base").value);
  if (!cur) return;
  const group = (e) => { const g = new Map(); for (const x of e?.samples || []) { const p = promptText(x.prompt); if (!g.has(p)) g.set(p, []); g.get(p).push(x); } return g; };
  const gc = group(cur), gb = base && base !== cur ? group(base) : null;
  const prompts = [...new Set([...gc.keys(), ...(gb ? gb.keys() : [])])];
  const draws = (list, other) => (list || []).map((x, i) => {
    const same = other && other.some((o) => o.reply === x.reply);
    return `<div class="draw${same ? " same" : ""}">${x.reply ? esc(x.reply) : "<i class=muted>(empty)</i>"}<span class="n">draw ${i + 1} · ${x.tokens ?? "?"} tokens${same ? " · identical in other column" : ""}</span></div>`;
  }).join("") || `<p class="empty">no sample for this prompt</p>`;
  $("#s-list").innerHTML = prompts.map((p) => `<div class="sample"><div class="p"><b>prompt</b> ${esc(p)}</div>
    <div class="sides${gb ? "" : " one"}">${gb ? `<div class="side"><h4>step ${base.step}</h4>${draws(gb.get(p), gc.get(p))}</div>` : ""}
    <div class="side"><h4>step ${cur.step}${cur.event === "done" ? " (final)" : ""}</h4>${draws(gc.get(p), gb?.get(p))}</div></div></div>`).join("");
}

function renderData() {
  const m = model();
  const c = m.counts, a = m.args;
  const extra = $("#data-extra");
  extra.replaceChildren();
  if (!c || typeof c !== "object") { $("#data-table").innerHTML = `<div class="empty" style="padding:.6rem .8rem">no data counts in the start record</div>`; return; }
  const rowsRaw = Object.entries(c).map(([src, v]) => ({ src, v: typeof v === "number" ? { train: v } : v || {} }));
  const totalTrain = rowsRaw.reduce((s, r) => s + (isNum(r.v.train) ? r.v.train : 0), 0);
  const cols = [["val", "val"], ["groups", "groups"], ["targets", "targets"], ["history_train", "with history"], ["gif_tagged_train", "gif-tagged"], ["duplicate_groups_dropped", "dup groups dropped"]].filter(([k]) => rowsRaw.some((r) => isNum(r.v[k])));
  const head = [{ t: "source", l: true }, { t: "train rows" }, { t: "share of train", title: "this source's train rows / all train rows (after repeats)" }, { t: "mix (config)", title: "the mix_* arg for this source" }, { t: "repeat" }, ...cols.map(([, t]) => ({ t }))];
  rowsRaw.sort((x, y) => (y.v.train || 0) - (x.v.train || 0));
  table($("#data-table"), head, rowsRaw.map(({ src, v }) => {
    const share = totalTrain && isNum(v.train) ? v.train / totalTrain : null;
    const [mixK, repK] = MIX_ARG[src] || [];
    return { cells: [src, isNum(v.train) ? v.train.toLocaleString() : "—",
      { html: share == null ? "—" : `${(share * 100).toFixed(1)}%<span class="share"><i style="width:${(share * 100).toFixed(1)}%"></i></span>` },
      mixK && isNum(a[mixK]) ? fmt(a[mixK], 3) : "—", repK && isNum(a[repK]) ? `×${a[repK]}` : "—",
      ...cols.map(([k]) => (isNum(v[k]) ? v[k].toLocaleString() : "—"))] };
  }).concat([{ cls: "hl", cells: ["total", totalTrain.toLocaleString(), "100%", "", "", ...cols.map(([k]) => { const s = rowsRaw.reduce((t, r) => t + (isNum(r.v[k]) ? r.v[k] : 0), 0); return s ? s.toLocaleString() : "—"; })] }]));

  // discord_vibe filter funnel
  const vb = c.discord_vibe;
  if (vb && isNum(vb.vibe_rows_scanned)) {
    const tot = vb.vibe_rows_scanned;
    const rej = Object.entries(vb).filter(([k, v]) => k.startsWith("vibe_rejected_") && isNum(v)).sort((x, y) => y[1] - x[1]);
    const rows = [["rows scanned", tot, "total"], ...rej.map(([k, v]) => [`rejected: ${k.slice(14).replace(/_/g, " ")}`, v, ""])];
    if (isNum(vb.vibe_skipped_seen)) rows.push(["skipped (already seen)", vb.vibe_skipped_seen, ""]);
    if (isNum(vb.vibe_groups_kept)) rows.push(["groups kept", vb.vibe_groups_kept, "kept"]);
    if (isNum(vb.groups)) rows.push(["groups after dedup", vb.groups, "kept"]);
    if (isNum(vb.train)) rows.push(["train rows", vb.train, "kept"]);
    const card = document.createElement("div");
    card.className = "card";
    card.innerHTML = `<h3>discord_vibe filter</h3><p class="cap" style="margin:0 0 .5rem">bars are a share of rows scanned; the vibe filter (min ${esc(a.discord_vibe_min ?? "?")}, min targets ${esc(a.discord_vibe_min_targets ?? "?")}) keeps ${isNum(vb.vibe_groups_kept) ? ((vb.vibe_groups_kept / tot) * 100).toFixed(2) + "%" : "?"} of scanned rows.</p>
      <div class="funnel">${rows.map(([l, v, cls]) => { const w = Math.max(0, Math.min(100, (v / tot) * 100)); return `<div>${esc(l)}</div><div class="lane" title="${esc(l)}: ${v.toLocaleString()} (${((v / tot) * 100).toFixed(2)}%)"><i class="${cls}" style="width:${w}%"></i></div><div class="num">${v.toLocaleString()}<br><span class="muted">${(v / tot * 100).toFixed(v / tot < 0.001 ? 3 : 1)}%</span></div>`; }).join("")}</div>`;
    extra.append(card);
  }
  // booper_local snapshot stats
  const bl = c.booper_local;
  if (bl && typeof bl === "object" && Object.keys(bl).some((k) => k.startsWith("local_"))) {
    const kv = (o) => Object.entries(o || {}).map(([k, v]) => `<dt>${esc(k.replace(/_/g, " "))}</dt><dd>${esc(typeof v === "number" ? v.toLocaleString() : v)}</dd>`).join("");
    const card = document.createElement("div");
    card.className = "card kv";
    card.style.display = "block";
    card.innerHTML = `<h3>booper_local snapshot</h3><dl>${kv({ "rows read": bl.local_rows, kept: bl.local_kept, "train rows": bl.train, "val rows": bl.val })}</dl>
      ${bl.local_rejected ? `<h3 style="margin-top:.5rem">rejected by reason</h3><dl>${kv(bl.local_rejected)}</dl>` : ""}
      ${bl.local_kept_by_signal ? `<h3 style="margin-top:.5rem">kept by signal</h3><dl>${kv(bl.local_kept_by_signal)}</dl>` : ""}`;
    extra.append(card);
  }
  // every other numeric count, per source
  const other = rowsRaw.filter(({ v }) => Object.keys(v).length > 1);
  if (other.length) {
    const card = document.createElement("div");
    card.className = "card";
    card.innerHTML = `<details><summary>all counts from the start record</summary><div class="tablewrap scroll"></div></details>`;
    const allKeys = [...new Set(other.flatMap(({ v }) => Object.keys(v)))].filter((k) => other.some(({ v }) => isNum(v[k])));
    table(card.querySelector(".tablewrap"), [{ t: "count", l: true }, ...other.map(({ src }) => src)], allKeys.map((k) => ({ cells: [k, ...other.map(({ v }) => (isNum(v[k]) ? v[k].toLocaleString() : ""))] })));
    extra.append(card);
  }
}

const CFG_GROUPS = [
  ["model / base", ["name", "config", "base", "device", "seq_len", "prompt_budget", "history_turns", "min_response", "expert_bucket", "export", "push"]],
  ["schedule", ["tokens", "examples", "lr", "warmup", "log_every", "eval_every", "ckpt_every", "seed", "resume", "smoke"]],
  ["batch", ["tokens_per_batch", "accum", "fixed_rows", "pad_multiple", "val_examples"]],
  ["gate", ["max_source_val_regression", "guard_sources", "guard_max_regression", "multiturn_must_improve"]],
  ["memory / throttle", ["grad_checkpoint", "grad_checkpoint_min_width", "mps_high_watermark", "mps_low_watermark", "mps_empty_cache_gb", "duty_cycle", "pause_on_battery"]],
];
function cfgVal(v) {
  if (v === null || v === undefined) return "—";
  if (Array.isArray(v)) return v.join(", ") || "[]";
  if (typeof v === "object") return JSON.stringify(v);
  if (typeof v === "number") return Math.abs(v) >= 1e5 ? v.toLocaleString() : String(v);
  return String(v);
}
function renderConfig() {
  const m = model(), a = m.args;
  const host = $("#cfg");
  if (!Object.keys(a).length) { host.innerHTML = `<p class="empty">no args in the start record</p>`; $("#cfg-all").innerHTML = ""; return; }
  const extra = { steps_total: m.steps_total, device: m.start?.device };
  const card = (title, pairs) => (pairs.length ? `<div class="card"><h3>${esc(title)}</h3><dl>${pairs.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(cfgVal(v))}</dd>`).join("")}</dl></div>` : "");
  let html = "";
  for (const [title, ks] of CFG_GROUPS) {
    const pairs = ks.filter((k) => k in a || (title === "model / base" && k === "device")).map((k) => [k, k === "device" ? (a.device ?? extra.device) : a[k]]);
    if (title === "schedule" && isNum(extra.steps_total)) pairs.splice(2, 0, ["steps_total", extra.steps_total]);
    html += card(title, pairs);
  }
  const mix = Object.keys(a).filter((k) => /^(mix_|repeat_)/.test(k) && isNum(a[k]) && a[k] !== 0);
  const mixPairs = mix.filter((k) => k.startsWith("mix_")).map((k) => { const src = k.slice(4); const rep = Object.keys(a).find((r) => r.startsWith("repeat_") && (r.slice(7) === src || (src === "norobots" && r === "repeat_norobots"))); return [k, `${a[k]}${rep && a[rep] !== 1 ? ` (×${a[rep]})` : ""}`]; });
  const vibe = Object.keys(a).filter((k) => k.startsWith("discord_vibe_") || k.startsWith("gif_") || k === "smoltalk_multiturn").map((k) => [k, a[k]]);
  html += card("mix (non-zero)", [...mixPairs, ...vibe]);
  host.innerHTML = html;
  table($("#cfg-all"), [{ t: "arg", l: true }, { t: "value", l: true }], Object.keys(a).sort().map((k) => ({ cells: [k, { t: cfgVal(a[k]), cls: "l" }] })));
}

function renderTables() {
  const m = model();
  const best = bestOf(m);
  const sm = smooth(m.train.map((r) => [r.step, r.loss]), "ema90");
  table($("#t-eval"), [{ t: "step", l: true }, "val", "train (EMA 0.9)", "gate", "time"], m.evals.map((e) => {
    const t = sm.filter((p) => p[0] <= e.step).pop();
    return { cls: best && best.step === e.step ? "hl" : "", cells: [`${e.step}${e.event === "done" ? " (final)" : ""}${best && best.step === e.step ? " ★ best" : ""}`, fmt(e.val, 4), fmt(t?.[1], 4),
      { t: e.source_gate === true ? "✓ pass" : e.source_gate === false ? "✕ fail" : "—", cls: e.source_gate === true ? "ok" : e.source_gate === false ? "no" : "" }, clock(e.t)] };
  }), { empty: "no evals yet" });
  table($("#t-train"), [{ t: "step", l: true }, "loss", "lr", "grad norm", "tok/s", "tok/s active", "tokens", "RSS GB", "idle s", "ETA", "time"],
    [...m.train].reverse().map((r) => ({ cells: [r.step, fmt(r.loss, 4), isNum(r.lr) ? r.lr.toExponential(3) : "—", fmt(r.grad_norm, 3), fmt(r.tok_s, 0), fmt(r.tok_s_active, 0), compact(r.tokens), fmt(r.rss_gb, 2), fmt(r.idle_s, 1), dur(r.eta_s), clock(r.t)] })), { empty: "no train rows yet" });
  const all = state.cache.get(state.run).recs.filter((r) => r.event);
  table($("#t-events"), [{ t: "event", l: true }, "time", "step", { t: "details", l: true }], all.map((e) => ({ cells: [e.event, clock(e.t), e.step ?? e.resumed_step ?? "—",
    { cls: "l", t: e.event === "start" ? `resumed_step ${e.resumed_step ?? 0} · steps_total ${e.steps_total ?? "?"} · ${e.device ?? ""}` : e.event === "done" ? `val ${fmt(e.val, 4)} · best step ${e.best_step ?? "—"} (${fmt(e.best_val, 4)}) · ${e.gated ? "gated" : "not gated"}` : `${e.reason ?? ""}${isNum(e.paused_s) ? ` · paused ${dur(e.paused_s)}` : ""}` }] })));
}

// ---------- compare ----------
function renderCmpPicker() {
  const host = $("#cmp-pick");
  const others = state.index.filter((r) => r.name !== state.run);
  host.innerHTML = `<label class="chk"><input type="checkbox" checked disabled> ${esc(state.run)} (current)</label>` +
    others.map((r) => `<label class="chk"><input type="checkbox" value="${esc(r.name)}"${state.cmp.includes(r.name) ? " checked" : ""}${!state.cmp.includes(r.name) && state.cmp.length >= 4 ? " disabled" : ""}> ${esc(r.name)}</label>`).join("");
  host.querySelectorAll("input[value]").forEach((el) => (el.onchange = () => {
    if (el.checked) state.cmp.push(el.value); else state.cmp = state.cmp.filter((n) => n !== el.value);
    renderCompare();
  }));
}
async function renderCompare(force = false) {
  renderCmpPicker();
  const names = [state.run, ...state.cmp];
  const host = $("#c-cmp");
  let entries;
  try { entries = await Promise.all(names.map((n) => loadRunData(n, force && n !== state.run))); }
  catch (e) { host.innerHTML = `<div class="empty">could not load: ${esc(e.message)}</div>`; return; }
  const metric = $("#cmp-metric").value;
  // colour follows the run: slot by its position in the selection, current run always slot 1.
  const series = entries.map((en, i) => {
    const mm = en.model;
    const pts = metric === "val" ? mm.evals.map((e) => [e.step, e.val]) : smooth(mm.train.map((r) => [r.step, r.loss]), "ema90");
    return { name: names[i], color: CMP_COLORS[i], pts, style: metric === "val" && pts.length < 3 ? "dots" : "line", endDot: metric === "val", dec: 4 };
  });
  lineChart(host, series, { height: 260, title: `compare ${metric}`, empty: "no data for this metric" });
}

// ---------- wiring ----------
$("#run").onchange = (e) => setRun(e.target.value);
$("#refresh").onclick = () => refresh(true);
$("#auto").onchange = () => { stampFetched(); schedule(); };
$("#smooth").onchange = () => renderTraining();
$("#fitY").onchange = () => renderTraining();
$("#earlier").onchange = () => renderTraining();
$("#s-step").onchange = () => renderSamples();
$("#s-base").onchange = () => renderSamples();
$("#cmp-metric").onchange = () => renderCompare();

function schedule() {
  clearInterval(state.timer);
  if ($("#auto").checked) state.timer = setInterval(() => { if (!document.hidden) refresh(); }, REFRESH_MS);
}
document.addEventListener("visibilitychange", () => { if (!document.hidden && $("#auto").checked && state.fetchedAt && Date.now() - state.fetchedAt > REFRESH_MS) refresh(); });
let rz, lastW = innerWidth;
addEventListener("resize", () => {
  if (innerWidth === lastW) return; // mobile URL-bar height changes don't need a redraw
  lastW = innerWidth;
  clearTimeout(rz);
  rz = setTimeout(() => { if (model()) { renderTraining(); renderEval(); renderCompare(); } }, 150);
});
addEventListener("scroll", hideTip, { passive: true });

refresh();
schedule();
