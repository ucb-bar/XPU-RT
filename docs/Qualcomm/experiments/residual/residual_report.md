# QRB5165 residual feedback — full experiment log

440 dispatches across 4 committed board traces (`runs/*/trace.csv`). Every row carries both the schedule's `predicted_duration_ms` and the board's `actual_start/end_ms`, so a run that already happened is a measurement of its own error.

## 1. The solo profile is biased, per backend, in opposite directions

| trace | n | co-runners | stalls >1 ms | stall total | HTA | DSP | CPU |
|---|---:|---:|---:|---:|---:|---:|---:|
| `v3_bundles` | 97 | 0 | 0 | 0 ms | 1.196 | — | 0.980 |
| `v3_bundles_dsp14_lazy` | 105 | 0 | 30 | 1316 ms | 1.044 | 0.975 | 1.001 |
| `v3_bundles_dsp9` | 97 | 0 | 0 | 0 ms | 1.082 | 0.978 | 0.994 |
| `v3_bundles_dsp_all_reset` | 141 | 0 | 5 | 347 ms | 1.090 | 0.888 | 1.154 |

Pooled: **CPU 0.999** (n=180, p10 0.89–p90 1.26), **DSP 0.907** (n=138, p10 0.81–p90 1.47), **HTA 1.082** (n=122, p10 0.94–p90 1.37).

HTA is under-estimated in every trace and DSP over-estimated in every trace it appears in. The DSP direction independently corroborates `docs/Qualcomm/qualcomm-qrb5165.md` §2, which measured the recorded DSP column ~16% pessimistic and suspected a slower host clock at capture time.

**There is no contention to learn from.** The `co-runners` column is zero everywhere: these workloads are serial chains, exactly as `docs/Qualcomm/qualcomm-qrb5165.md` §3 reports. So the residual measured here is *calibration bias*, not co-runner interference — a distinction that matters, because only the latter would depend on what else is scheduled.

## 2. Feeding the error back works — within one configuration

| trace | logerr before | after | after 2nd round | MAE before | after |
|---|---:|---:|---:|---:|---:|
| `v3_bundles` | 0.1108 | 0.0875 | 0.0875 | 2.887 | 2.539 |
| `v3_bundles_dsp14_lazy` | 0.0420 | 0.0308 | 0.0308 | 1.229 | 1.146 |
| `v3_bundles_dsp9` | 0.0820 | 0.0844 | 0.0844 | 2.321 | 2.202 |
| `v3_bundles_dsp_all_reset` | 0.1269 | 0.0605 | 0.0605 | 2.354 | 1.254 |

Mean logerr **0.0904 → 0.0658 (27.2% reduction)**. The second round moves it almost nowhere: one round of feedback reaches the fixpoint, because the correction is a single multiplicative constant per backend and applying it twice would double-count.

## 3. It does not transfer across configurations

Leave-one-trace-out: fit on the other three, score the held-out one.

| held-out | n | logerr before | after | verdict |
|---|---:|---:|---:|---|
| `v3_bundles` | 97 | 0.1108 | 0.1017 | improves |
| `v3_bundles_dsp14_lazy` | 105 | 0.0420 | 0.0609 | **worse** |
| `v3_bundles_dsp9` | 97 | 0.0820 | 0.0828 | **worse** |
| `v3_bundles_dsp_all_reset` | 141 | 0.1269 | 0.1126 | improves |

Helps on **2/4** held-out traces.

The four traces are the same network under four *runtime* configurations (eager budget-9, lazy budget-14 + LRU evict, all-DSP with backend reset). `dsp14_lazy` already predicts well (logerr 0.042) and a correction fit elsewhere makes it worse. So the bias is a property of the configuration that ran, not of the silicon — which is why a board-level constant is the wrong shape for it, and why the feedback artifact must be keyed by configuration.

Excluding the 35 stall-delayed dispatches does not change this (still 2/4), so residency stalls are not the cause either.

## 4. Consequences for scheduling

* A correction fit on the run you are about to repeat is worth ~27% of the estimate error; one fit on a different configuration is not worth applying.
* The stall term is the largest single dynamic effect and is entirely configuration-borne: `dsp14_lazy` loses **1316 ms** across 30 stalls and `dsp_all_reset` **347 ms** across 5, while the other two lose nothing. No per-kernel cost model can carry that; it belongs to context residency.
* Contention could not be evaluated at all, because no committed QRB5165 trace has two dispatches in flight at once. The contention sweep predicts concurrent schedules but has never been run on the board — that run is the missing measurement.

