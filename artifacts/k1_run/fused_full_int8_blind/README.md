# fused_full, pure int8 — kept as evidence, not as a deployable profile

This is the profile and graph of the configuration whose fuse layer requantizes
the camera and ToF branches to all-zero. It is preserved because it is the
measurement behind the finding, and because the numbers are otherwise
indistinguishable from a working model:

| input into the fuse       | scale_in | ratio to scale_out | max int8 -> levels |
|---------------------------|----------|--------------------|--------------------|
| vision_fc (camera, 512ch) | 0.001922 | 0.001116           | 0.142  -> 0        |
| depth_fc  (ToF, 64ch)     | 0.004530 | 0.002629           | 0.334  -> 0        |
| lowdim    (state, 21ch)   | 1.723068 | 1.000000           | 127.000            |

`lowdim`'s `optical_flow` component reaches 218.8, and the concat takes its
output scale from that same max-abs, so the other two branches lose every level.

Measured on the K1: feeding zeros for both sensors gives bit-identical commands
over six consecutive real gate-course frames. `max_abs_err=0` against the golden
holds, because the kernels are correct — they are faithfully executing a network
that has lost 95% of its input. Against fp32 over 12 real frames the command
error is rel L2 0.2385 (hybrid: 0.0768); cosine similarity is 0.9711, which is
why cosine is not the metric to report for this model.

Total service time 3.851 ms, median of 6 warm reps, 13 dispatches.

DO NOT cite this as fused_full's cost. The authoritative profile is the hybrid
(int8 encoders, fp16 fuse/LSTM/head), which needs GCC >= 14 for its Zvfh kernels
— see MODELBLASTER_KERNEL_CC in harness_xpurt_linux/CMakeLists.txt.
