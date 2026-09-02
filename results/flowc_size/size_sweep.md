# Does network size decide which feedback path pays?

Two axes, because they answer different questions. Across networks the op count spans three orders of magnitude but topology and op mix vary too, so a correlation there would be suggestive at best. The ViNT observation encoder measured at three batch sizes is the controlled version: same graph, same ops, only the tensors grow.

## Across networks

| network | ops | baseline ms | best ms | speedup | winning knob |
|---|---:|---:|---:|---:|---|
| `mlp_control` | 7 | 0.0295 | 0.0255 | 1.16x | `+slice` |
| `dronet` | 29 | 7.4655 | 0.6509 | 11.47x | `+backend` |
| `fused_full` | 91 | 3.4339 | 0.4524 | 7.59x | `+precision` |
| `yolov8n` | 233 | 72.7216 | 25.2316 | 2.88x | `+backend` |
| `vint` | 1925 | 59.7749 | 23.2007 | 2.58x | `+slice` |

**Size does not predict the knob.** Ordered by op count the winners run `+slice`, `+backend`, `+precision`, `+backend`, `+slice` — no monotone relationship, and the largest network (vint, 1925 ops) and the smallest (mlp_control, 7 ops) share a winner while everything between them differs. The obvious hypothesis, that big graphs have room to slice and small ones are all overhead, is not what the measurements say.

What decides it instead is **what the op set can compile to**: dronet's 11.47x is one backend move, available because its ops run on the DSP; vint has to be cut before any accelerator will take it at all; fused_full's win is a precision change its ops happen to reward. Those are properties of the op mix, not of the size.

## Within one network — ViNT observation encoder, batch 1/2/3

| variant | ops | dsp@int8 | cpu@int8 | cpu@fp32 | best |
|---|---:|---:|---:|---:|---|
| `vint_obs_b1` | 512 | 4.6896 | 6.064 | 13.7658 | `dsp@int8` |
| `vint_obs_b2` | 512 | 5.3222 | 5.8871 | 25.8323 | `dsp@int8` |
| `vint_obs_b3` | 512 | 5.3422 | 8.2126 | 33.8325 | `dsp@int8` |

**The accelerator's advantage GROWS with size.** Three times the batch costs the DSP 1.14x but CPU fp32 2.46x, so the DSP's margin over CPU fp32 widens from 2.94x at batch 1 to 6.33x at batch 3. The placement knob is therefore worth more on bigger tensors even though the CHOICE of knob does not track size across networks — the two findings are about different things, and only this one is controlled.

## Not measurable host-side

| variant | why it is missing |
|---|---|
| `yolov8_nano_64x96` | yolov8n at 64x96 input; exists on the K1/spacemit tree, no QRB5165 cells |
| `yolov8_nano_128x192` | yolov8n at 128x192 input; exists on the K1/spacemit tree, no QRB5165 cells |
| `dronet_small` | dronet with reduced channel width; exists on the K1/spacemit tree, no QRB5165 cells |

`board_plan.sh` builds and profiles these under the shared board lock. Until it runs, the resolution axis is untested on this board and nothing here should be read as covering it.

