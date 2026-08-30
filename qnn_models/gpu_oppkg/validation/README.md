# Validation inputs and raw outputs

Everything the accuracy tables in `../README.md` were computed from. All dumps
are raw int8 (`--use_native_input_files --use_native_output_files`), so the two
sides of every comparison saw byte-identical inputs.

| directory | what |
|---|---|
| `inputs/` | fixed inputs: `obs_{i}.raw` fp32 and `obs_q_{i}.raw` int8 (8 seeded vectors), `dronet_q{i}.raw` (5 real calibration frames), `front_grey_q0.raw` (vision_conv tile), `obs_f16/f32.raw` |
| `out/out_gpu_fix`, `out/out_dsp_ref`, `out/out_native_Cpu` | mlp_control int8, 8 inputs, final output — GPU (this package) / DSP / CPU |
| `out/out_cpu_dbg`, `out/out_gpu_dbg`, `out/out_c77` | mlp_control per-tensor dumps, including the constant-activation run used to isolate the per-layer kernels |
| `out/d_cut2_gpu`, `out/d_cut2_dsp` | the FullyConnected+ELU two-op graph, full 256-element activation |
| `out/dump_vc_{gpu,dsp,cpu}` | fused_split vision_conv tile (4 int8 convolutions) |
| `out/dbg_dr_{gpu,cpu,dsp}` | dronet int8, every intermediate tensor, one frame |
| `out/o5_{gpu,cpu,dsp}` | dronet int8 final outputs over 5 real frames |
| `out/dump_f{16,32}_gpu_{flowc,stock}` | float kernels, ours vs the stock GPU package |

Reproduce the numbers:

```bash
python3 ../tools/emulate_mlp_int8.py --compare out/out_gpu_fix
python3 ../tools/compare_dumps.py out/dbg_dr_gpu/Result_0 out/dbg_dr_cpu/Result_0 \
        --label-a GPU --label-b CPU --order ../model/dronet_ref_net.json
```
