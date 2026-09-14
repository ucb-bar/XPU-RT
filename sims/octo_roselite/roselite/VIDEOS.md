# RoSE-lite latency study — video deliverables

Eleven clips in `videos/`, all 1760×990 h264. Seven are one-per-case rollouts
with a title card and a burned-in caption; **four** are the animated
hardware-lane gantt composited with a rollout on a shared time axis.

**Two of the four gantt replays are SERIAL and two are genuinely OVERLAPPING**,
and every one of them recomputes its own lane overlap from the trace it draws
and prints it on the frame:

| gantt replay | schedule | MEASURED lane overlap | max concurrent lanes | instances in flight |
|---|---|---|---|---|
| `gantt_283.4ms_serial_3way-CPU-DSP-HTA` | serial | **0.00 ms (0.0%)** | 1 | 1 |
| `gantt_684.8ms_serial_CPU-only-monolith` | serial | **0.00 ms (0.0%)** | 1 | 1 |
| `gantt_pipelined_200ms_MEASURED` | **pipelined** | **55.5 ms (5.1%)** | **2** | **2** |
| `gantt_pipelined_110ms_MEASURED` | **pipelined** | **694.2 ms (62.3%)** | **3** | **4** |

The two pipelined replays draw multi-instance walks **actually executed on the
QRB5165**, coloured by Octo instance. They are not the modelled pipelined
strategy the rollout clips use — see
[Task 3](#task-3--the-measured-pipelined-replays).

Every number below is tagged **MEASURED** or **MODELLED**. Read
`RESULTS.txt` for the study, `latency_eval.py` for the timing model, and
`/scratch2/dima/misc_sw/XPU-RT/qnn_models/octo/OCTO_INT8_QRB5165.md` for the
board measurements.

---

## Framing, on every frame of every clip

* The int8 QRB5165 pipeline is a **PERFORMANCE VEHICLE**. The 3-way chain
  measures **cos 0.008** against the JAX golden (`OCTO_INT8_QRB5165.md` §6).
  The policy in every rollout here ran at **full fp32 on the host GPU**;
  the only thing the board contributes is **when** each action chunk becomes
  available. No frame claims the int8 model's outputs drove the robot.
* **Serial vs overlapping is stated per clip, never assumed.** The two Task 2
  gantt replays are SERIAL: MEASURED `max_concurrent_lanes = 1`, **0.00 ms**
  overlap, and neither draws two concurrent inferences because neither trace
  contains any. The two Task 3 replays are genuinely OVERLAPPING, from walks
  executed on the board, and print their real **5.1%** / **62.3%** overlap.
* In the seven **rollout** clips, **"pipelined" is a MODELLED control
  strategy** — a fresh inference issued every 200 ms — and each says so on the
  title card and on every frame. That labelling is correct for those clips:
  their timing model is modelled, not replayed from a pipelined trace. The
  Task 3 gantt replays are the measured counterpart, and are labelled MEASURED.
* **Superseded claim, corrected.** The Task 1 and Task 2 clips were made when
  no pipelined Octo schedule had ever run on the board, and some of their text
  says so. That is **no longer true**: pipelined walks have since been executed
  and measured on the QRB5165 (Task 3). The rollout clips' own wording remains
  accurate about *their* timing model; only the blanket claim that the hardware
  has never run a pipelined schedule is out of date.
* The harness is **not reproducible**: identical command and seed gave stock
  13/24 then 15/24 (`RESULTS.txt` §0). Each clip is *a* representative episode,
  never *the* episode, and the caption names the run directory and episode
  index it came from.

## Where each latency number comes from (all MEASURED on the QRB5165)

| latency | configuration | source |
|---|---|---|
| 0 ms | no latency — the stock free-running configuration | — |
| **283.4 ms** | 3-way CPU+DSP+HTA int8, ungated | median of the 20 `[iter] … wall=` lines in `repro_runs/ungated_20260905-131037.log` (283.36 ms) |
| **555 ms** | numerically-valid **fp32** Octo path, checkpoint 1.0 | `OCTO_INT8_QRB5165.md` §1 |
| **684.8 ms** | CPU-only int8 monolith, ungated | median of the 20 `[iter] … wall=` lines in `repro_runs/mono_20260905-130719.log` (684.82 ms) |

`D = ceil(latency / 200 ms)` is **MODELLED**: `PutEggplantInBasketScene-v0` sets
`control_freq = 5`, so one env step is 200 ms.

---

## Task 1 — one clip per highlighted case

Structure: 4 s title card, then the rollout at **real time** (one env step =
200 ms), then a 1.8 s tail. The caption panel carries the latency, the
schedule, D, the arm's MEASURED success rate with n, and this episode's
outcome, on every frame.

"arm SR / n" is the **published** figure from `RESULTS.txt` over the full
sweep. "recording run" is the small pool this clip was picked from — it differs
from the published SR because n is small and the harness is nondeterministic;
both are given so nothing is overstated.

| clip | len | latency | schedule | D | arm SR (MEASURED) | n | this episode | source episode |
|---|---|---|---|---|---|---|---|---|
| `case_0ms_baseline_ensembler-on_SUCCESS.mp4` | 14.4 s | 0 ms | free-running, stock ensembler **ON** | 0 | **52.8%** | 72 | **SUCCEEDED** | `runs_video/base_0ms_ens-stock` ep02 (recording run 7/12) |
| `case_283.4ms_pipelined_SUCCESS.mp4` | 14.2 s | 283.4 ms | **pipelined** (MODELLED) | 2 | **56.9%** | 72 | **SUCCEEDED** | `runs_video/lat283_pipelined` ep05 (recording run 8/12) |
| `case_283.4ms_serial_FAILURE.mp4` | 30.0 s | 283.4 ms | serial | 2 | **15.8%** | 120 | **FAILED** | `runs_video/lat283_serial` ep00 (recording run 3/12) |
| `case_555ms_serial_FAILURE.mp4` | 30.0 s | 555 ms | serial | 3 | **9.7%** | 72 | **FAILED** | `runs_video/lat555_serial` ep04 (recording run 2/12) |
| `case_555ms_pipelined_FAILURE.mp4` | 30.0 s | 555 ms | **pipelined** (MODELLED) | 3 | **20.8%** | 72 | **FAILED** | `runs_video/lat555_pipelined` ep04 (recording run 4/8) |
| `case_684.8ms_serial_FAILURE.mp4` | 30.0 s | 684.8 ms | serial | 4 | **3.3%** | 120 | **FAILED** | `runs_video/lat684_serial` ep03 (recording run 1/12) |
| `case_684.8ms_pipelined_DEAD_never-moves.mp4` | 30.0 s | 684.8 ms | **pipelined** (MODELLED) | 4 | **0.0%** | 8 | **FAILED — the robot never moves** | `runs_video/lat684_pipelined` ep00 (recording run 0/6) |

### What each clip shows

**`case_0ms_baseline_ensembler-on_SUCCESS.mp4`** — the reference. A fresh
inference every 200 ms, the stock `ActionEnsembler` averaging the last 4
predictions, and every action acting on the observation that produced it.
A clean 42-step pick-and-place. *Representative SUCCESS: this arm is 52.8%, so
about half its episodes fail — that is stated on the frame.*

**`case_283.4ms_pipelined_SUCCESS.mp4`** — the headline. 283.4 ms of MEASURED
board latency costs nothing: **56.9% vs the 52.8% baseline, Fisher p = 0.74,
n.s.** Two predictions still target each step, so the timestep-aligned average
survives. The frame states that pipelined is MODELLED, not a hardware
schedule. *Representative SUCCESS.*

**`case_283.4ms_serial_FAILURE.mp4`** — the contrast, and the deployable one.
Same 283.4 ms, one inference in flight: **15.8% vs the pipelined 56.9%,
p = 6.5e-09**. The clip shows the modal failure of this arm — it grasps the
eggplant (61.7% of episodes do) and then cannot place it (only 15.8% reach the
target). *Representative FAILURE: 84% of this arm's episodes fail.*

**`case_555ms_serial_FAILURE.mp4`** — the grasp itself starts to go. Grasp rate
falls from 61.7% at 283.4 ms to 47.2% here, success to 9.7%. The chosen episode
never achieves a grasp, which is now the majority outcome (53%).
*Representative FAILURE.*

**`case_555ms_pipelined_FAILURE.mp4`** — even the optimistic strategy breaks.
At D = 3 only **one** prediction still targets the current step, so the
averaging that rescued 283.4 ms is gone: **20.8% vs 52.8% baseline,
p = 1.2e-04**. *Representative FAILURE.*

**`case_684.8ms_serial_FAILURE.mp4`** — the CPU-only int8 monolith. One
observation every 4 control steps; **3.3%** success and only **26.7%** of
episodes manage a grasp at all. The arm flails near the eggplant.
*Representative FAILURE.*

**`case_684.8ms_pipelined_DEAD_never-moves.mp4`** — **structurally dead, and
the stillness is the result.** At D = 4 a 4-entry chunk is already stale when
it lands, so **zero** predictions target the current step, the timestep-aligned
average is empty, and `latency_eval.py` falls through to the zero-delta HOLD
forever. Verified two ways: `sum |applied action| = 0.000` over all 120 steps
in **all six** recorded episodes, and the decoded rollout's maximum mean
per-pixel deviation from frame 0 is **0.05/255** (h264 noise) against **19.3**
for a moving episode. Nothing was cherry-picked here — every episode of this
arm is this clip.

### No success is presented as typical

Two clips show a success (the 52.8% and 56.9% arms, where a success is roughly
the modal outcome). Both carry the arm's SR on the frame. **No success is shown
from a low-success arm** — the 15.8%, 9.7%, 20.8%, 3.3% and 0% arms are all
represented by failures, which is what those arms mostly do.

---

## Task 2 — the animated hardware-lane gantt (SERIAL traces)

`animate_octo_gantt.py`. Style and lane colours match
`XPU-RT/qnn_models/octo/plot_octo_gantt.py` (the static version of the same
trace); mechanics — lane rows, playhead, bars revealed as the playhead passes,
per-frame compositing, mp4 writing — follow
`XPU-RT/sims/smolvla_demo/animate_schedule_gantt.py`.

Everything is on **one shared env wall-clock axis** in ms. Four panels: the
rollout advancing at 200 ms per env step; a status panel; the hardware-lane
gantt; and an "applied action" strip showing, per control step, whether the
robot is holding (`HOLD`) or which chunk entry (`e0`…`e3`) it is executing.
The first env steps play at 0.22x so the mechanism is legible (9 steps in the
283.4 ms clip, 12 in the 684.8 ms one, which has a longer cycle), then the
playback ramps to real time.

Bars are the **MEASURED** per-dispatch trace, read from the runtime's
`AGENTS_QNN_TRACE_BEGIN/END` CSV block. Header quirk handled in the parser:
`kind` holds the machine id (`CPU_X`/`CPU_E`/`CPU_P`) and `backend_label` holds
the segment name; the lane comes from `actual_backend`.

| clip | len | trace (MEASURED) | dispatches | one pass | lane busy | overlap |
|---|---|---|---|---|---|---|
| `gantt_283.4ms_serial_3way-CPU-DSP-HTA.mp4` | 35.2 s | `repro_runs/ungated_20260905-112925.log` | 71 | 250.07 ms | DSP 110.4 / CPU 67.8 / HTA 41.1 ms | **0.00 ms** |
| `gantt_684.8ms_serial_CPU-only-monolith.mp4` | 37.4 s | `repro_runs/mono_20260905-130719.log` | 21 | 677.08 ms | **CPU 675.5 (99.8%) / DSP 0.0 / HTA 0.0** | **0.00 ms** |

### What the animation makes visible

* **When each inference is issued and when its result lands.** A red `▼ issue
  sK` at the issue time, a green `▲ lands sK+D` D steps later, and a connector
  between them. Serial: the next request is issued the instant the previous
  lands, so issues fall on steps 0, D, 2D, …
* **Which chunk entry is being applied in between.** The strip under the gantt.
  At D = 2 it reads `e0, e1, e0, e1, …` — **entries e2 and e3 are never
  reached**, because a new chunk supersedes the old one after 2 steps. At
  D = 4 all four entries are consumed. The footer states which case applies.
* **The robot holding before the first result arrives.** Steps 0…D−1 are
  orange `HOLD` cells and the banner reads `ROBOT HOLDING`; the rollout image
  is genuinely static there.
* **The ceil() quantisation.** Compute finishes at 250 ms (3-way) but the env
  cannot consume the result until the 400 ms control tick. That gap is drawn as
  a shaded "ready, waiting for the 200 ms tick" band. The status line
  distinguishes the three states: `COMPUTING`, then `this trace done at 250 ms;
  median 283`, then `done — held for the 200 ms control tick`.
* **Why the CPU-only config dies** (the 684.8 ms clip). One grey `trunk` bar
  plus 20 `score` dispatches occupy the CPU lane for **675.5 of the 677.1 ms
  pass — 99.8%** — while **DSP and HTA sit at exactly 0.0 ms**. That 677 ms
  consumes 85% of the 800 ms chunk horizon that D = 4 allows, so there is no
  room left to shorten the control loop. The 3-way clip beside it does the same
  work in 250 ms by spreading it over three lanes.

### One honest caveat about the 283.4 ms gantt

The **283.4 ms** that sets `D = 2` is the median wall over 20 iterations of
`ungated_20260905-131037.log`. The per-dispatch trace **drawn** is from
`ungated_20260905-112925.log`, whose own iteration spans **250.07 ms** (that
run's median is 263.08 ms). Both are the same 3-way ungated configuration, and
the difference sits inside the documented ~15% between-process noise floor
(`OCTO_INT8_QRB5165.md` §5.4). The animation shows both numbers — the trace
span in the panel title, the 283.4 ms median as a dashed marker and in the
status line — rather than reconciling them silently. Nothing is rescaled.

Also MODELLED, and said on the frame: the same measured trace is **redrawn once
per issue**. The board was not run 60 times in lockstep with the simulator.
### Both Task 2 replays were re-emitted

They were rebuilt from the same traces, episodes and arguments after Task 3
landed, so all four gantt replays come from one codebase and share one style.
The rebuild changed exactly one thing on the frame: the serial footer used to
read *"No pipelined Octo schedule has ever been executed on the board"*, which
Task 3 falsified. It now reads *"This trace draws one inference at a time"* —
a claim about the trace being drawn, which stays true. Everything else, the
0.00 ms overlap included, is unchanged and recomputed.

### No gantt for 555 ms

555 ms is the fp32 Octo path figure from `OCTO_INT8_QRB5165.md` §1, not a
per-dispatch trace, so there is nothing measured to animate. Its two rollout
clips are in Task 1.

---


---

## Task 3 — the MEASURED PIPELINED replays

The two Task 2 replays are both **serial**. These two are the overlapping
counterpart: multi-instance walks **actually executed on the QRB5165**, drawn
with `animate_octo_gantt.py --schedule pipelined`, coloured by Octo instance so
the overlap is visible rather than asserted.

| clip | len | trace (MEASURED) | dispatches | instances | issue cadence | per-inference | completes every | lane overlap | max lanes | in flight |
|---|---|---|---|---|---|---|---|---|---|---|
| `gantt_pipelined_200ms_MEASURED.mp4` | 19.4 s | `repro_runs/pipe200_gated_20260905-164357.log` | 355 | 5 | 200 ms | **260 ms** | 203 ms | **55.5 ms (5.1%)** | **2** | **2** |
| `gantt_pipelined_110ms_MEASURED.mp4` | 19.4 s | `repro_runs/pipe110x10_20260905-165246.log` | 710 | 10 | 110 ms | **385 ms** | 98 ms | **694.2 ms (62.3%)** | **3** | **4** |

Throughput, MEASURED over repeated iterations of the same walk:
**200 ms x 5** — wall median 1159.2 ms over n=6 → 231.8 ms/inference, 4.31 inf/s.
**110 ms x 10** — wall median 1176.6 ms over n=20 → 117.7 ms/inference, 8.50 inf/s.

Both carry a same-session **SERIAL control**
(`repro_runs/ungated_20260905-164735.log`, 71 dispatches), whose overlap is
recomputed by the same code and printed on the frame: **0.00 ms, 0.0%, max 1
lane**. The contrast is measured on both sides.

### `gantt_pipelined_200ms_MEASURED.mp4` — the headline

The 56.9% pipelined arm assumes a fresh inference every 200 ms. The board now
**demonstrably sustains it**: completions land every **203 ms** (median of 4
gaps, 173–220), against a 200 ms control tick. Per-inference latency is
**260 ms**, so `D = 2` and an action applied at step *k* was computed from
frames captured at steps *k−3* and *k−2* — **600 and 400 ms** of observation
staleness, stated on the frame.

Paired with `runs_video/lat283_pipelined ep05`, the episode underlying
`case_283.4ms_pipelined_SUCCESS.mp4`, because that is the arm whose 56.9% this
illustrates.

### `gantt_pipelined_110ms_MEASURED.mp4` — the throughput ceiling

**8.50 inf/s, 62.3% overlap, all three lanes concurrently busy, four instances
in flight.** It is also where the distinction the figure exists to make bites:

* **The env still steps at 200 ms.** At a 110 ms issue cadence the hardware
  produces roughly two inferences per control step, and the animation counts
  them: **34 of 75 inferences re-read a camera frame an earlier one had already
  read.** The camera yields one frame per 200 ms step, so those add throughput,
  not information. They are drawn as **hollow `dup` markers** on the hardware
  row, distinct from the solid `reads sK` markers.
* **Latency gets worse, not better.** Contention raises per-inference latency
  from 260 → **385 ms**. `D` is still 2 — but only by 15 ms, and the animation
  says so: **1 of 10 instances exceeded 400 ms** and would have landed at
  `D = 3`, where this arm's SR falls to 20.8%.
* **So the control side is unchanged.** Same `D`, same arm, same 56.9%. The
  clip reuses the *same* rollout episode as the 200 ms replay deliberately, and
  says "same arm as the 200 ms walk — D unchanged at 2" on the frame, rather
  than implying more throughput bought a better robot.

### What `--schedule pipelined` does differently

Every number it prints is recomputed from the trace being drawn; nothing is
hardcoded into the figure.

* **Bars coloured by Octo instance**, same palette as
  `octo_pipe/plot_measured_lanes.py`, so instance overlap is visible.
* **Lane concurrency is counted in DISTINCT LANES**, not dispatches. On these
  traces each lane serialises (verified: max same-lane concurrency = 1), so the
  two agree — but the animation tiles the walk onto an exact issue grid, which
  *can* double-book a lane, and a per-dispatch count would then report more
  "lanes" than the machine has.
* **Two marker rows, because there are two clocks.** Lower row = the hardware
  axis, one mark per inference the board starts, labelled with the camera frame
  it consumed (`reads s3`, or hollow `dup s3`). Upper row = the control axis,
  one mark per 200 ms env step, labelled `sK→sK+D`.
* **An "instances in flight" band** under the lane rows, plus a live per-frame
  readout of which instances are on the board and how many lanes are busy.
* **A D-margin check**, counting how many MEASURED inferences exceeded
  `D × 200 ms` and would have landed a step later.

The applied-action strip follows `latency_eval.py --pipeline` exactly: the
action at step *t* is the timestep-aligned mean over every delivered chunk
entry targeting *t* — issues in `[t−3, t−D]`, so `e3+e2` at `D = 2`.

### The one thing that is MODELLED here

The walk is redrawn once per issue on an **exact** issue grid, while the board's
own releases drifted (200 ms walk: 236, 223, 209, 204 ms apart; 110 ms walk: 97,
107, 105, 97, 98, 98, 98, 100, 100 ms). The grid therefore does not reproduce
the walk's overlap exactly, and **both numbers are printed** — the walk's
MEASURED 5.1% / 62.3% and the drawn composite's 8.7% / 36.1%. Neither is
rescaled to match the other. As in Task 2, the board was not run once per
simulator step.


## Regenerating

```bash
bash record_cases.sh     # re-record the episode pools -> runs_video/
bash make_all_videos.sh  # rebuild all 11 clips in videos/ from those + the traces
```

`make_all_videos.sh` builds Task 1 (7 rollout clips), Task 2 (2 serial gantt
replays) and Task 3 (2 pipelined gantt replays). All board traces it reads live
in `XPU-RT/qnn_models/octo/repro_runs/` and are produced by
`reproduce_octo_int8.sh`; the pipelined ones come from the `/root/qnn_runtime_pipe`
(200 ms x 5) and `/root/qnn_runtime_pipe110x10` (110 ms x 10) runtimes on the
board, with `/root/qnn_runtime5` as the serial control.

Board traces are **not** bit-reproducible either: re-running the same pipelined
walk gave 4.6% then 5.1% overlap, and 57.6% then 62.3%, which is the documented
~15% between-process noise floor (`OCTO_INT8_QRB5165.md` §5.4). Every figure
recomputes its numbers from the trace it is handed, so a re-run relabels itself
rather than silently disagreeing with its own caption.

`record_cases.sh` writes to **`runs_video/`**, not `runs/`, on purpose:
`analyze.py` globs `runs/lat*_from_arrival_ens-none_rng*`, so recording into
`runs/` would silently fold extra episodes into the published `RESULTS.txt`
aggregation.

Re-running `record_cases.sh` will **not** reproduce the same episodes — the
harness is nondeterministic. `make_all_videos.sh` pins the episode index per
case, so the clips are reproducible from the recordings that exist.

Scripts:

* `make_case_video.py` — one captioned case clip from one recorded episode.
* `animate_octo_gantt.py` — the composited hardware-lane gantt animation;
  `--schedule serial` (Task 2) and `--schedule pipelined` (Task 3).

Environment: conda `octo_sim`, `VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json`
(SAPIEN otherwise picks llvmpipe and dies with `ErrorExtensionNotPresent`),
`XLA_PYTHON_CLIENT_PREALLOCATE=false` — all set by `env.sh`.

## Also in `videos/`

`lat283.4ms_serial_rng0_ep*.mp4` — four pre-existing raw 640×480 clips from the
283.4 ms serial arm, no caption and no title card. Superseded by
`case_283.4ms_serial_FAILURE.mp4`; kept because they are the untouched
recorder output.
