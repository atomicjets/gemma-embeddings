# save as bench_truncate.py
import numpy as np, time, json, os

N = int(os.getenv("N", "200000"))     # vectors to simulate
D = int(os.getenv("D", "768"))        # original dim
D2 = int(os.getenv("D2", "256"))      # target dim
B = int(os.getenv("BATCH", "4096"))   # batch size

np.random.seed(0)

def trunc_norm(x, d2=256):
    y = x[:, :d2].copy()                               # slice/copy
    n = np.linalg.norm(y, axis=1, keepdims=True) + 1e-12
    y /= n
    return y

# generate one batch to warm-up allocator & BLAS
warm = np.random.randn(B, D).astype(np.float32)
_ = trunc_norm(warm, D2)

t0 = time.perf_counter()
done = 0
while done < N:
    n = min(B, N - done)
    x = np.random.randn(n, D).astype(np.float32)
    y = trunc_norm(x, D2)
    done += n
t1 = time.perf_counter()
vecs_per_s = N / (t1 - t0)
print(f"truncate+renorm throughput: {vecs_per_s:,.0f} vectors/sec (D={D}→{D2})")

# JSON serialization size & speed (optional)
x = np.random.randn(B, D).astype(np.float32)
y = trunc_norm(x, D2)
t2 = time.perf_counter()
_ = json.dumps(y.tolist())           # simulate Qdrant JSON payload
t3 = time.perf_counter()
print(f"JSON serialize 256-d {B} vecs: {(t3-t2)*1000:.1f} ms")

t4 = time.perf_counter()
_ = json.dumps(x.tolist())           # 768-d for comparison
t5 = time.perf_counter()
print(f"JSON serialize 768-d {B} vecs: {(t5-t4)*1000:.1f} ms")
