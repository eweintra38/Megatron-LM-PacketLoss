"""NumPy mirror of the DROP_LOG_INTERVAL logging in param_and_grad_buffer.py.

Mirrors one bucket group on N ranks: owners update their shards with uncorrelated
random-walk steps, broadcasts and gradient pieces are lost i.i.d. with probability p,
and the per-rank accumulation, the restore order, the pair-drift all-reduce identity
and the JSON records follow the implementation. Writes JSONL files that can be fed to
tools/packet_loss_summary.py.

Usage: python drop_log_mirror.py <out_prefix> [p] [iterations] [interval] [repeats]   (Python 3.8+, NumPy)
"""
import json
import sys

import numpy as np

prefix = sys.argv[1]
P = float(sys.argv[2]) if len(sys.argv) > 2 else 0.4
T = int(sys.argv[3]) if len(sys.argv) > 3 else 4000
INTERVAL = int(sys.argv[4]) if len(sys.argv) > 4 else 100
REPEATS = int(sys.argv[5]) if len(sys.argv) > 5 else 3
N, SH = 8, 64


def run(seed, path):
    rng = np.random.default_rng(seed)
    owner = np.zeros((N, SH))                         # owners' shards
    copies = np.zeros((N, N, SH))                     # copies[i, k] = rank i's copy of shard k
    prev = copies.copy()                              # per-rank backup (= copy after last sync)
    age = np.zeros((N, N), int)
    host = dict.fromkeys(["age_sum", "copies", "updates", "S_sum", "inv_S_sum", "S_count",
                          "grad_lost", "grad_links", "param_lost", "param_links", "pair_shards"], 0.0)
    dev = dict.fromkeys(["stale_err_sum", "update_sq_sum", "pair_drift_sum"], 0.0)
    max_identity_err = 0.0
    with open(path, "w") as out:
        for t in range(1, T + 1):
            # --- gradient side: S at each owner (own piece always counts)
            keep_g = rng.random((N, N)) >= P
            np.fill_diagonal(keep_g, True)
            S = keep_g.sum(axis=0)
            host["grad_links"] += N * (N - 1)
            host["grad_lost"] += (~keep_g).sum()
            host["S_sum"] += S.sum(); host["inv_S_sum"] += (1.0 / S).sum(); host["S_count"] += N
            # --- owner update and all-gather (fresh values everywhere)
            owner += rng.standard_normal((N, SH))
            fresh = np.broadcast_to(owner, (N, N, SH)).copy()
            # --- per-rank drop/restore with logging (t == 1 is the first sync: no logging)
            first = t == 1
            pair_step = not first and t % INTERVAL == 0
            keep_p = rng.random((N, N)) >= P
            np.fill_diagonal(keep_p, True)
            new = fresh.copy()
            for i in range(N):
                for k in range(N):
                    gap = fresh[i, k] - prev[i, k]
                    if k != i:
                        age[i, k] = 0 if keep_p[i, k] or first else age[i, k] + 1
                    if not first:
                        if k == i:
                            dev["update_sq_sum"] += (gap ** 2).sum(); host["updates"] += 1
                        else:
                            host["param_links"] += 1; host["copies"] += 1; host["age_sum"] += age[i, k]
                            if not keep_p[i, k]:
                                host["param_lost"] += 1; dev["stale_err_sum"] += (gap ** 2).sum()
                    if keep_p[i, k] or first:
                        prev[i, k] = fresh[i, k]
                    else:
                        new[i, k] = prev[i, k]
            copies = new
            if pair_step:
                e = copies - fresh                         # e[i, k] = rank i's copy - fresh shard k
                own_sq = (e ** 2).sum(axis=2).sum(axis=0)   # all-reduce of per-shard ||e_i||^2
                sum_sq = (e.sum(axis=0) ** 2).sum(axis=1)   # ||sum_i e_i||^2 per shard
                m = N - 1
                pair = (m * own_sq - sum_sq) / (m * (m - 1) / 2)
                # brute force: mean over non-owner pairs of ||copy_i - copy_k||^2
                for j in range(N):
                    others = [i for i in range(N) if i != j]
                    brute = np.mean([((copies[a, j] - copies[b, j]) ** 2).sum()
                                     for x, a in enumerate(others) for b in others[x + 1:]])
                    max_identity_err = max(max_identity_err, abs(brute - pair[j]))
                dev["pair_drift_sum"] += pair.sum(); host["pair_shards"] += N
            if t % INTERVAL == 0:
                s = {**host, **dev}
                ratio = lambda a, b: s[a] / s[b] if s[b] else None
                out.write(json.dumps({
                    "iteration": t, "param_syncs": t, "interval": INTERVAL, "N": N,
                    "param_loss_rate": ratio("param_lost", "param_links"),
                    "grad_loss_rate": ratio("grad_lost", "grad_links"),
                    "mean_age": ratio("age_sum", "copies"), "stale_err": ratio("stale_err_sum", "copies"),
                    "update_sq": ratio("update_sq_sum", "updates"),
                    "pair_drift": ratio("pair_drift_sum", "pair_shards"),
                    "mean_S": ratio("S_sum", "S_count"), "mean_inv_S": ratio("inv_S_sum", "S_count"),
                }) + "\n")
                host = dict.fromkeys(host, 0.0); dev = dict.fromkeys(dev, 0.0)
    return max_identity_err


if __name__ == "__main__":
    for r in range(1, REPEATS + 1):
        err = run(r, f"{prefix}_r{r}.jsonl")
        print(f"repeat {r}: pair-drift identity vs brute force, max abs difference {err:.2e}")
    p = P
    print(f"theory (N={N}, p={p}): mean age {p/(1-p):.4f} | owner-copy {p/(1-p):.4f} | "
          f"pair {2*p/(1-p*p):.4f} | variance factor {(1-p**N)/(1-p):.4f}")
