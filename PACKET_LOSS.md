# Packet-loss simulation for data-parallel training

This fork of [HabanaAI/Megatron-LM](https://github.com/HabanaAI/Megatron-LM) (release 1.21.0)
adds a simulation of lost messages in the two data-parallel collectives of the distributed
optimizer: the gradient reduce-scatter and the parameter all-gather. It is the code for
*Distributed Training under Packet Loss* (E. Weintraub, R. Banner, A. Orda, 2026).

All changes are in `megatron/core/distributed/param_and_grad_buffer.py` and
`examples/llama/pretrain_llama.sh`. Model code and optimizers are unchanged.

## What is simulated

Each of the N data-parallel ranks owns one shard of the parameters (ZeRO-style).
Per iteration:

| Step (paper, Sec. 3) | Implementation |
|---|---|
| (1) Each rank computes gradients on its own, possibly stale, copy of the model | unchanged Megatron forward/backward on the rank's local parameters |
| (2) Gradient piece i → j is lost with probability p; owner j averages what arrived | before the reduce-scatter, a sender zeroes each slice it "drops"; the drop masks are all-reduced to get the survivor count S_j, and the owner's shard is rescaled from 1/N to 1/S_j. A rank's own slice is never dropped, so S_j ≥ 1 |
| (3) Owner j updates its shard | unchanged distributed optimizer step |
| (4) Shard j → rank i is lost with probability p; rank i keeps its old copy | after the all-gather, a receiver restores each "dropped" shard to the last copy it received, so staleness accumulates over consecutive losses |

Every rank draws from its own random stream, derived from `DROP_SEED` and its global rank.
With the i.i.d. model, drop decisions are independent across links, iterations and ranks; with
the Gilbert-Elliott model, losses come in bursts (see below).

## Settings

Set these before calling `examples/llama/pretrain_llama.sh`; the script forwards them to all ranks.

| Variable | Default | Meaning |
|---|---|---|
| `EW_PACKET_LOSS_PROB_GRAD` | `0.0` | drop probability of each gradient slice (sender → owner) |
| `EW_PACKET_LOSS_PROB_PARAM` | `0.0` | drop probability of each parameter shard (owner → receiver) |
| `HL_DROP_MODEL` | `uniform` | `uniform` (i.i.d. Bernoulli) or `ge` (Gilbert-Elliott) |
| `EW_DROP_SEED` | `0` | seed of the drop streams; use a different value for each repeated run |
| `EW_GE_P`, `EW_GE_R` | `0.01`, `0.10` | Gilbert-Elliott transition probabilities G→B and B→G |
| `EW_GE_PG`, `EW_GE_PB` | `0.01`, `0.70` | drop probability in the Good and Bad state |
| `EW_DROP_LOG_INTERVAL` | `0` | log the measurements below every N iterations (`0` = off) |
| `EW_DROP_LOG_FILE` | `<output dir>/drop_metrics.jsonl` | where global rank 0 appends the measurements |
| `HL_SEED` | `1234` | Megatron's `--seed` (model initialization and data order) |
| `HL_LR`, `HL_MIN_LR` | model preset | peak and minimum learning rate |
| `HL_LR_WARMUP_ITERS` | model preset | learning-rate warmup iterations |

Example (one node of 8 Gaudi 3, a 134M-parameter model, 40% loss on both collectives,
repeat #1, measurements every 100 iterations):

```bash
HL_NUM_NODES=1 HL_DP=8 HL_TP=1 HL_PP=1 \
HL_DATA_DIR_ROOT=/path/to/red_pajama HL_TOKENIZER_MODEL=/path/to/tokenizer.model \
HL_LLAMA_VER=2 HL_LLAMA_MODEL_SIZE=7 HL_NUM_LAYERS=12 HL_HIDDEN_SIZE=768 HL_NUM_HEADS=12 \
HL_NUM_QUERY_GROUPS=12 HL_FFN_HIDDEN_SIZE=2048 HL_SEQ_LEN=1024 HL_GBS=256 HL_MICRO_BATCH=8 \
HL_TRAIN_ITERS=10000 HL_LR=6e-4 HL_MIN_LR=6e-5 HL_LR_WARMUP_ITERS=200 HL_SEED=1235 \
EW_PACKET_LOSS_PROB_GRAD=0.4 EW_PACKET_LOSS_PROB_PARAM=0.4 EW_DROP_SEED=1 EW_DROP_LOG_INTERVAL=100 \
examples/llama/pretrain_llama.sh
```

## Measurements

With `EW_DROP_LOG_INTERVAL > 0`, each line of the log file summarizes one interval. All values
are per shard (one rank's slice of the parameter buffer) and averaged over ranks:

| Field | Meaning |
|---|---|
| `mean_age` | staleness age (iterations since the last received broadcast) of the non-owned copies |
| `stale_err` | `‖copy − owner's shard‖²` of the non-owned copies after the broadcast (0 if received) |
| `update_sq` | `‖Δθ‖²` of each rank's own shard per iteration, an estimate of `σ_j²` |
| `pair_drift` | mean `‖copy_i − copy_k‖²` over pairs of non-owners, measured at the end of the interval |
| `mean_S`, `mean_inv_S` | number of gradient pieces that arrived for each rank's own shard, and its inverse |
| `param_loss_rate`, `grad_loss_rate` | realized fractions of lost broadcasts and gradient pieces |

Staleness errors and update sizes are measured right after the all-gather, before a lost shard
is restored. The pair drift uses one all-reduce of the staleness errors per interval. Logging
also works with drops disabled (a lossless baseline). `tools/packet_loss_summary.py` turns one
or more log files (repeats) into early- and late-window averages, normalized by `update_sq`:

```bash
python tools/packet_loss_summary.py run1/drop_metrics.jsonl run2/drop_metrics.jsonl run3/drop_metrics.jsonl
```

## Checking the analysis

`tools/packet_loss_validation/` holds the NumPy scripts behind Appendix A of the thesis. They
need only Python 3.8+ and NumPy, not Megatron or accelerators, and use fixed seeds.

- `appendix_checks.py` checks the results of Section 3: exact enumeration and Monte Carlo of
  the drift theorem (uncorrelated, constant, mixed and momentum updates), the aggregation
  weights and variance factor, the full protocol with SGD on least squares (drift and the
  staleness bias bound), Gilbert-Elliott and position-dependent losses with and without
  interleaving, and shard-level versus packet-level loss. Run all checks with
  `python tools/packet_loss_validation/appendix_checks.py`, or a subset, e.g. `... A1 A3`.
- `drop_log_mirror.py` re-implements the logging above on N simulated ranks and writes log
  files in the same format, to check the estimators (including the pair-drift identity):

```bash
python tools/packet_loss_validation/drop_log_mirror.py /tmp/mirror 0.4 4000 100 3
python tools/packet_loss_summary.py /tmp/mirror_r1.jsonl /tmp/mirror_r2.jsonl /tmp/mirror_r3.jsonl
```

## Requirements and limitations

- Needs `--use-distributed-optimizer` (the script's default) and synchronous collectives: do not
  enable `--overlap-grad-reduce` or `--overlap-param-gather`, and keep a single
  distributed-optimizer instance. The code stops with an error otherwise.
- With overlap disabled each buffer is a single bucket, so every link gets exactly one drop
  decision per iteration and collective.
- The unit of loss is a whole shard slice (tens of MB for a billion-parameter model), not a
  network packet: `p` is a per-shard loss rate.
- Gilbert-Elliott: each rank keeps one Good/Bad chain per collective, advanced once per
  iteration. Given the state, each of the rank's links is dropped independently, so a Bad
  period hits all of the rank's links at once (like a congested NIC). For gradients the chain
  belongs to the sender (its outgoing slices), for parameters to the receiver (its incoming
  shards). With the defaults, Good periods last about 100 iterations and Bad periods about 10
  iterations at 70% loss; the mean loss is about 7.3%. (There is one chain per bucket group; a
  dense model with overlap disabled has a single group.)
- The drop streams and Gilbert-Elliott states are not saved in checkpoints. After a resume
  they restart from `EW_DROP_SEED` (and the Good state); change the seed when resuming if
  fresh draws are wanted.

## Cloning on Windows

Two upstream files have a `:` in their path (`requirements/pytorch:24.01/`, `pytorch:24.07/`),
which Windows does not allow. Clone without checkout and exclude them:

```bash
git clone --no-checkout https://github.com/eweintra38/Megatron-LM-PacketLoss.git
cd Megatron-LM-PacketLoss
git config core.protectNTFS false
git sparse-checkout set --no-cone '/*' '!/requirements/pytorch:24.01/' '!/requirements/pytorch:24.07/'
git checkout main
```
