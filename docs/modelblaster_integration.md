# ModelBlaster ↔ XPU-RT: the two feedback channels

XPU-RT schedules; ModelBlaster compiles and runs. They talk through files, and
there are **two channels, not one** — a distinction that is easy to miss because
both are called "feedback".

| channel | says | file | consumed by |
|---|---|---|---|
| **compile advice** | how to REWRITE the graph | `compile_advice.json` | `advice_to_*_hint.py` → `apply_*_hint.py` |
| **runtime feedback** | how to PLACE and SIZE what is already there | `xpurt_feedback.json` | `emit_compile_advice.py --feedback` |

The first changes what the dispatches ARE — fuse, split, unfuse, shard, choose
an implementation. The second leaves the graph alone and says a dispatch ran
slower than predicted, or has slack, or should be pinned. A rewrite needs a
reprofile and a fresh verdict; a placement hint does not.

Both are **inert by default**. With neither file present, every existing
invocation behaves exactly as it did.

## What this replaced

This document used to describe merlin. XPU-RT and merlin collaborated through
`scripts/merlin_adapter.py`, an `ingest_xpurt_feedback` MCP tool, and a
`TelemetrySink` in `scheduler_runner.cc`.

Two of those three never existed. The MCP tool was not in the merlin checkout,
and no runner emitted the telemetry format `streaming_feedback.py` tailed —
merlin's `scheduler_runner.cc` writes a trace CSV. The adapter was real and was
the only caller of `xpu-rt/feedback.py`. All three are retired along with the
merlin submodule; what follows is what actually runs.

## Channel 1 — compile advice (rewrite the graph)

```
gen_mb/profile/**/results.csv          measured, per dispatch, per topo tag
        │
        ▼  scripts/emit_compile_advice.py
compile_advice.json                    {fuse, split, unfuse, shard,
        │                               choose_implementation}
        ▼  scripts/advice_to_<verb>_hint.py
<verb>_hint.json                       a contract ModelBlaster accepts
        │
        ▼  ModelBlaster/pipeline/apply_<verb>_hint.py
graph.json (rewritten)
        │
        ▼  verify bit-exact → board → reprofile → re-solve
scripts/compare_candidates.py          accept / reject, and WHICH TERM decided
```

Each bridge enforces every constraint the rewriter enforces, in the bridge,
where the advice that caused a refusal is still in hand. Emitting a hint the
rewriter refuses is the failure these files exist to prevent.

`shard` is the odd one out and worth reading `advice_to_shard_hint.py` for: it
does not rewrite the graph at all. It annotates one dispatch with a core width,
so the dispatch count, ids and edges are unchanged. Everything else in this
table is a graph rewrite.

## Channel 2 — runtime feedback (place and size)

Two ways in, same file out.

**Batch**, from the solve itself:

```bash
python scripts/run_xpurt_schedule.py --networks-json <spec>.json --emit-feedback
# -> schedules/xpurt_feedback.json
```

This lives in the solver rather than in a script that reads the written
schedule back, and that is deliberate. `feedback.derive_dispatch_hints` wants
the solver's `(t, alpha)` arrays; reconstructing `alpha` from a serialized
schedule means inferring a one-hot assignment from a machine label, and any
dispatch whose label is ambiguous silently becomes a hint about the wrong
combination.

**Streaming**, from the board while the run is still going:

```bash
# build the walker with the emitter
MB_XPURT_STREAM=1 bash ModelBlaster/scripts/run_xpurt_k1.sh <schedule.json> \
    | tee /tmp/xpurt.jsonl

python xpu-rt/streaming_feedback.py \
    --telemetry-stream /tmp/xpurt.jsonl \
    --out schedules/xpurt_feedback.json \
    --windows-from data/toplevel/<spec>.json \
    --run-id k1_$(date -u +%Y%m%dT%H%M%SZ) --follow
```

The walker's trace block is printed when the run ENDS, which is enough to
explain a run and useless for responding to one: a schedule that starts missing
deadlines in instance 3 of 60 keeps missing them for the other 57. The stream is
the same numbers, available while there is still something to do about them.

### The telemetry line

One JSON object per dispatch **end**, from `generate_xpurt_main.py` under
`-DMODELBLASTER_XPURT_STREAM`:

```json
{"entry_id":7,"network":"dronet","instance":2,"dispatch_id":13,
 "impl":"ime","hart":3,"predicted_start_ms":1.25,"predicted_duration_ms":0.5,
 "start_ticks":1000,"end_ticks":13000}
```

**Ticks are `rdtime` at 24 MHz**, not core cycles and not microseconds:
`rdcycle` SIGILLs from userspace on this board, so the harness reads `rdtime`,
whose device-tree `timebase-frequency` is 24000000.

It is written with **one `write()` per line, not `printf`**. Several workers
reach the emitter concurrently and interleaved `printf` calls tear a line into
fragments that are not JSON.

### What the board does not know

The walker has no idea what its deadlines are — periods and `window_duration`
live in the workload spec, not in the binary. So it emits no `deadline_miss`,
and there is no skip mechanism, so no `skip_fired`.

Without `--windows-from`, the miss rate is reported as **unknown, not zero**.
This matters: a network whose name ended in a digit once had its instance index
misparsed, its deadline became ~48 s, and it could never miss. The reported zero
was structural and read exactly like a pass. A structural zero that looks like a
measurement is the failure mode this channel is shaped to avoid.

The signals derivable from the stream alone are therefore about the **cost
model** — measured duration against predicted — which is what drives
`prefer_finer` / `prefer_coarser`.

## The return edge: who reads `xpurt_feedback.json`

`scripts/emit_compile_advice.py --feedback <xpurt_feedback.json>`.

For a while nothing did, which made channel 2 a producer with no consumer —
the same shape as the problem the shard chain was written to fix, pointing the
other way.

**The consumer is the advice producer, not ModelBlaster directly**, and that
is forced rather than chosen. The obvious move is a `feedback_to_hints.py`
beside the five `advice_to_*_hint.py` bridges. It cannot be written honestly:

| hint | what a hint file would have to invent |
|---|---|
| `prefer_finer` | the split factor `ceil(service / slot)` — no slot budget in the feedback |
| `consider_fuse_with_pred` | the *group* of dispatches to fuse — needs the graph |
| `pin_target=<x>` | `x` is a machine combination, not a kernel implementation |

Only `emit_compile_advice` holds both the graph and the periodic budget. A
bridge that guessed them would be inventing exactly the numbers the loop
exists to measure.

So the measured run **corroborates or contradicts** advice derived from
profiles, and never manufactures it:

* an item the run agrees with gains confidence (`medium` → `high`);
* an item the run contradicts is demoted to `unchanged`, with the reason in
  the rationale and `demoted_by_measurement` in the evidence;
* everything is recorded either way, so a reader can see *why* an item's
  confidence is what it is — and can undo the judgement by ignoring the field.

The contradiction table is deliberately **not** the complement of the
corroboration one. `prefer_coarser` contradicts splitting, but `prefer_finer`
does not contradict fusing: a dispatch can be both slower than predicted and
worth fusing with its neighbour, and treating the hints as opposites would
suppress correct advice on exactly the dispatches under most pressure.

Feedback is keyed per **instance** (`dronet7_dispatch_3`) and advice is per
**dispatch**, so instances are unioned — a dispatch that earned a hint in any
instance carries it. Union rather than majority because these hints already
survived `streaming_feedback`'s own rate thresholds; requiring a majority
would be filtering twice with the second filter undocumented.

The report distinguishes two things that are easy to conflate:

```
feedback (k1_2026...): 1 corroborated, 0 contradicted,
                       5 reported-on but unrelated, 23 not reported on
```

"the run said nothing about this dispatch" and "the run said something that
does not bear on this recommendation" are different facts, and only the first
means the measurement missed it.

## Hint vocabulary

Closed set, target-agnostic, shared by both entry points:

| hint | fires when |
|---|---|
| `prefer_coarser` | slack available; the dispatch is much faster than predicted |
| `prefer_finer` | ran slower than predicted, or idle gaps suggest unexploited parallelism |
| `consider_fuse_with_pred` | cross-cluster transfer dominates the duration |
| `pin_target=<name>` | this op is much faster on one combination |
| `consider_split_backend` | the current target is the slow side |

## Merge semantics

`xpurt_feedback.json` is rewritten on every streaming post, **merging by
set-union on hints per dispatch, keyed on `run_id`**. That is what makes it safe
to call repeatedly through a long run: a dispatch that earned `prefer_finer` in
instance 4 does not lose it because instance 40 was quiet. A *different*
`run_id` starts fresh — a new campaign must not inherit the last one's
conclusions.

## Related

* [`the_loop.md`](the_loop.md) — the whole cycle and which script owns each arrow
* [`k1_cost_by_pred.md`](k1_cost_by_pred.md) — what `consider_fuse_with_pred` is
  reasoning about, measured
* [`k1_contention.md`](k1_contention.md) — the null result, and why co-runner
  placement is not the mechanism
