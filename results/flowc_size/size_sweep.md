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

## Yolo resolutions and dronet variants (K1 / spacemit_x60)

These are the size variants QRB5165 has no cells for. Different silicon, so they do not transfer to the Qualcomm numbers above — but they are the only MEASURED size axis for these two networks, and they answer the question directly.

| variant | ms | vs the smallest |
|---|---:|---:|
| `yolov8_nano_64x96.int8` | 47.729 | 1.00x |
| `yolov8_nano_128x192.int8` | 193.918 | 4.06x |
| `yolov8_nano.int8` | 226.865 | 4.75x |
| `dronet.int8` | 8.334 | 1.00x |
| `dronet.split_x2.int8` | 9.769 | 1.17x |
| `dronet.split_x4.int8` | 12.403 | 1.49x |

**Yolo scales with pixels, not with anything cleverer.** 64x96 -> 128x192 is 4x the input area and 4.06x the time; the full resolution adds a further 1.17x. Nothing about the cost is sublinear, so there is no size at which the network suddenly becomes cheap to run whole.

**Splitting dronet always costs.** 1.00x whole, 1.17x at split_x2, 1.49x at split_x4 — monotonically worse. That is the same verdict the QRB5165 ladder reached from different measurements on different silicon: dronet is 0.66 ms there and 8.3 ms here, and in both cases it is too small to repay a cut. Two independent confirmations of one rule.

## Multi-hart scaling vs size — the K1 feedback path itself

This is the knob the K1 loop actually turns, so how it behaves across sizes is the closest available analogue to the question asked of QRB5165.

| variant | 1 hart | 2 | 4 | 8 | best | knee |
|---|---:|---:|---:|---:|---:|---:|
| `dronet.int8` | 8.334 | 6.053 | 5.247 | 5.320 | 1.59x | 4 harts |
| `ffn_block.int8` | 26.611 | 15.648 | 9.713 | 7.715 | 3.45x | 8 harts |
| `yolov8_nano_64x96.int8` | 47.729 | 29.562 | 24.353 | 23.948 | 1.99x | 8 harts |

**The sharding path's value is a property of the network, not of its size.** ffn_block reaches 3.45x and is still improving at 8 harts; yolov8_nano_64x96 manages 1.99x; dronet gets 1.59x and SATURATES at 4, getting slower at 8. yolo_64x96 is 5.7x larger than dronet yet scales only marginally better, while ffn_block — between them in cost — scales twice as well as either. So even for the one knob K1 does turn, size does not predict how much it is worth.

## Not measurable host-side

| variant | why it is missing |
|---|---|
| `yolov8_nano_64x96` | yolov8n at 64x96 input — MEASURED on K1/spacemit above; no QRB5165 cells |
| `yolov8_nano_128x192` | yolov8n at 128x192 input — MEASURED on K1/spacemit above; no QRB5165 cells |
| `dronet_small` | dronet with reduced channel width — MEASURED on K1/spacemit above; no QRB5165 cells |

`board_plan.sh` builds and profiles these on QRB5165 under the shared board lock. The size axis itself is answered above from K1 measurements; what remains open is only whether the SAME size behaviour holds on Qualcomm silicon, which is a question about transfer between targets rather than about size.

