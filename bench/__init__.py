"""Re-runnable performance benchmarks for babble on CPU.

`bench_inference.py` is the whole thing: TTFT, TPS (cached and uncached decode),
cold-vs-warm, a torch thread-count sweep, RAM and CPU while it runs, and the
training cost of one pretrain run. It writes `BENCHMARKS.md`-shaped numbers so a
change to the model or the decode path can be re-measured instead of guessed at.
"""
