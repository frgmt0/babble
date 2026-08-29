// booper.frgmt.xyz — static assets plus the /api/runs feed behind /runs.
// POST /api/runs/<name>   (Authorization: Bearer $RUNS_TOKEN)  append one metrics record
// GET  /api/runs          list runs (name, last update, last step)
// GET  /api/runs/<name>   all records for a run
// Storage: KV binding RUNS, key run:<name> -> JSON array, key index -> {name: meta}.
const MAX_RECORDS = 5000;
const json = (body, status = 200, extra = {}) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json", "cache-control": "no-store", ...extra },
  });

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const m = url.pathname.match(/^\/api\/runs(?:\/([A-Za-z0-9._-]{1,64}))?$/);
    if (!m) return env.ASSETS.fetch(request);
    const name = m[1];

    if (request.method === "GET" && !name) {
      const index = (await env.RUNS.get("index", "json")) || {};
      return json(Object.values(index).sort((a, b) => b.updated - a.updated));
    }
    if (request.method === "GET") {
      const recs = (await env.RUNS.get(`run:${name}`, "json")) || [];
      return json(recs);
    }
    if (request.method === "POST" && name) {
      const auth = request.headers.get("authorization") || "";
      if (!env.RUNS_TOKEN || auth !== `Bearer ${env.RUNS_TOKEN}`) return json({ error: "unauthorized" }, 401);
      let rec;
      try {
        rec = await request.json();
      } catch {
        return json({ error: "bad json" }, 400);
      }
      const key = `run:${name}`;
      const recs = (await env.RUNS.get(key, "json")) || [];
      recs.push(rec);
      // Keep the array bounded: drop the oldest plain train-loss rows first.
      while (recs.length > MAX_RECORDS) {
        const i = recs.findIndex((r) => "loss" in r && !("val" in r));
        recs.splice(i >= 0 ? i : 0, 1);
      }
      await env.RUNS.put(key, JSON.stringify(recs));
      const index = (await env.RUNS.get("index", "json")) || {};
      index[name] = {
        name,
        updated: Date.now(),
        step: rec.step ?? index[name]?.step ?? 0,
        steps_total: rec.steps_total ?? index[name]?.steps_total ?? null,
        done: rec.event === "done" || index[name]?.done || false,
      };
      await env.RUNS.put("index", JSON.stringify(index));
      return json({ ok: true, n: recs.length });
    }
    return json({ error: "method" }, 405);
  },
};
