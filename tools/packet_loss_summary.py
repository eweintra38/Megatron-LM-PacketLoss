"""Summarize the packet-loss metrics logged with EW_DROP_LOG_INTERVAL > 0.

Usage:
  python tools/packet_loss_summary.py run1/drop_metrics.jsonl [run2/... run3/...]
         [--early 0.1] [--late 0.5]

For each file, records are split into an early window (first fraction --early of the
records) and a late window (last fraction --late). Drift quantities are normalized by
the mean update size of the same window (sigma_j^2). With several files (repeats), the
mean and standard deviation across files are printed.
"""
import argparse
import json
import statistics


def load(path):
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return sorted(rows, key=lambda r: r["param_syncs"])


def mean(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return sum(vals) / len(vals) if vals else None


def window_stats(rows):
    upd = mean(rows, "update_sq")
    inv_s = mean(rows, "mean_inv_S")
    n = rows[0]["N"]

    def norm(key):
        v = mean(rows, key)
        return v / upd if v is not None and upd else None

    return {
        "mean age (iterations)": mean(rows, "mean_age"),
        "owner-copy error / sigma^2": norm("stale_err"),
        "pair drift / sigma^2": norm("pair_drift"),
        "variance factor N*E[1/S]": n * inv_s if inv_s is not None else None,
        "parameter loss rate": mean(rows, "param_loss_rate"),
        "gradient loss rate": mean(rows, "grad_loss_rate"),
    }


def fmt(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return "--"
    if len(vals) == 1:
        return f"{vals[0]:.4g}"
    return f"{statistics.mean(vals):.4g} +- {statistics.stdev(vals):.2g}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("files", nargs="+")
    ap.add_argument("--early", type=float, default=0.1, help="fraction of records in the early window")
    ap.add_argument("--late", type=float, default=0.5, help="fraction of records in the late window")
    a = ap.parse_args()

    per_window = {"early": [], "late": []}
    for path in a.files:
        rows = load(path)
        k_early = max(1, round(len(rows) * a.early))
        k_late = max(1, round(len(rows) * a.late))
        per_window["early"].append(window_stats(rows[:k_early]))
        per_window["late"].append(window_stats(rows[-k_late:]))

    print(f"{len(a.files)} run(s); early = first {a.early:.0%}, late = last {a.late:.0%} of the records")
    keys = list(per_window["early"][0])
    width = max(len(k) for k in keys)
    print(f"{'':{width}}  {'early':>20}  {'late':>20}")
    for k in keys:
        early = fmt([s[k] for s in per_window["early"]])
        late = fmt([s[k] for s in per_window["late"]])
        print(f"{k:{width}}  {early:>20}  {late:>20}")


if __name__ == "__main__":
    main()
