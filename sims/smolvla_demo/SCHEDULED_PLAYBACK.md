# SmolVLA at QRB5165 timing, played back in Isaac sim time

`run_demo.sh` runs SmolVLA in the IsaacLab Arena `gr1_microwave` task as if
inference were free: every env step calls `select_action`, and a fresh action
chunk appears the instant the queue empties. On the board it does not.

`run_demo_scheduled.sh` plays the profiled QRB5165 schedule back in sim time,
the same way `sims/scripts/play/play_dronet_mlp_scheduled.py` does for the
DroNet+MLP stack in the warehouse/forest envs. An observation is captured at the
sim time inference *starts*; the chunk it produces lands `makespan` ms of sim
time later; the robot moves on stale actions in between. Local wall-clock is
deliberately not what drives it — the policy runs on whatever GPU is present,
and the board timing sets the cadence.

## The numbers that make it interesting

From `schedules/scheduled_networks_smolvla_v3_unrolled10_qrb5165_greedy_profiled.json`
(102 dispatches; `metadata.profile_hw` maps the scheduler's machines onto
HTA/DSP/CPU):

    makespan          3351.7 ms      vision 1083.6 / prefill 583.8 / decode 1496.1
    chunk_size        50 actions     (= n_action_steps, so the whole chunk runs)
    control rate      30 Hz          -> 33.3 ms per env step
    chunk covers      50 x 33.3      = 1666.7 ms of motion
    latency           3351.7/33.3    = 100.6 env steps

**One inference costs twice the motion its chunk covers.** No scheduling fixes
that; it bounds the duty cycle:

| replan policy | cadence | steady-state stalled |
|---|---|---|
| `on_empty` (think only once the chunk runs out) | covers + latency = 5018 ms | **66.8 %** |
| `pipelined` (one inference always in flight) | latency = 3352 ms | **50.3 %** |

Pipelining is worth ~16 points of duty cycle and is the cheaper of the two
things you can do. The other is to make the chunk cover more time or the
inference cost less.

## Measured, 5 episodes per mode

    mode                    lat ms   succ  steps med  stall meas  stall ss
    none                         0    5/5         49        0.0%      0.0%
    schedule/on_empty         3352    4/5         51       11.5%     66.8%
    schedule/pipelined        3352    4/5         55       17.8%     50.3%

**Read this task's success numbers with care.** `gr1_microwave` finishes in a
median of 49 env steps — just under one 50-action chunk — so most episodes end
before the first chunk is even exhausted and never exercise the latency at all.
That is why `stall meas` (11-18 %) is far below the steady-state figure: it is
averaged over episodes that mostly stop early. The 4/5 vs 5/5 is five episodes
and is not a claim about success rate; the episodes that failed are exactly the
ones that needed a second chunk. A longer-horizon task would sit at the
steady-state numbers, and that is the figure to design against.

## Running it

    ./run_demo_scheduled.sh                                  # board timing, on_empty
    ./run_demo_scheduled.sh --replan pipelined               # one inference in flight
    LATENCY_MODE=none ./run_demo_scheduled.sh                # free-inference baseline
    EPISODES=5 MAX_STEPS=300 ./run_demo_scheduled.sh
    python3 summarise_scheduled.py                           # the table above
    python3 plot_scheduled.py --out ../../plots              # Gantt + loop timeline

`--latency-mode manual --latency-ms N` sweeps a hypothetical latency, which is
the way to ask "what would this cost if the expert work got 2x faster".

## Video

`--video` writes `runs/scheduled/video_<mode>_ep<n>.mp4` at the control rate, so
one second of video is one second of sim time. Each frame carries a status band
-- blue while a fresh chunk is executing, red and reading STALLED while one is
in flight -- and a strip underneath that accumulates the duty cycle over the
episode, so the stall pattern is visible as it builds rather than only in the
summary. The env's own recorder is left off; frames come from the POV camera
observation and are annotated here.

To put the modes side by side (they end at different steps, so the shorter ones
hold their last frame):

    cd runs/scheduled
    ffmpeg -i video_none_ep0.mp4 -i video_schedule_on_empty_ep0.mp4 \
           -i video_schedule_pipelined_ep0.mp4 -filter_complex "\
    [0]tpad=stop_mode=clone:stop_duration=10,trim=duration=3.2,setpts=PTS-STARTPTS[a];\
    [1]tpad=stop_mode=clone:stop_duration=10,trim=duration=3.2,setpts=PTS-STARTPTS[b];\
    [2]tpad=stop_mode=clone:stop_duration=10,trim=duration=3.2,setpts=PTS-STARTPTS[c];\
    [a][b][c]hstack=inputs=3[v]" -map "[v]" -r 30 -pix_fmt yuv420p video_compare_3up.mp4

## Two things worth knowing before you re-run

* **The first chunk is primed.** With latency applied to it there is no
  commanded pose to hold yet, so the stall path commands zero joint targets and
  the GR1 simply folds up — that measures the harness, not the deployment.
  `--no-prime` restores the naive behaviour if you want to see it.
* **`terminated` is not success.** This env has two termination terms,
  `success` and `time_out`, and both raise `terminated`, so the obvious
  `terminated.any()` scores a timeout as a win. The driver reads
  `info["final_info"]["is_success"]`.

Isaac writes its log to `$TMPDIR/isaaclab/logs`; on a shared host that collides
with whoever ran it first, so the runner points `TMPDIR` somewhere private.
