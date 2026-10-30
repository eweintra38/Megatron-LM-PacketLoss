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

Example (8 nodes × 8 Gaudi 3, 40% loss on both collectives, repeat #1):

```bash
HL_NUM_NODES=8 HL_DP=64 HL_TP=1 HL_PP=1 HL_HOSTSFILE=/path/to/hostsfile \
HL_DATA_DIR_ROOT=/path/to/red_pajama HL_TOKENIZER_MODEL=/path/to/tokenizer.model \
EW_PACKET_LOSS_PROB_GRAD=0.4 EW_PACKET_LOSS_PROB_PARAM=0.4 EW_DROP_SEED=1 \
examples/llama/pretrain_llama.sh
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
