# Runnable examples

One directory per topic, one self-contained script per idea. Every script

* runs from the repo root with no arguments,
* prints what it did and why, not just that it finished,
* is honest about what it cannot do here — anything needing the board says so
  and stops, rather than printing a number it did not measure.

```bash
.venv/bin/python examples/run_all.py            # the CI-cheap subset
.venv/bin/python examples/solvers/compare_solvers.py
```

| directory | what it shows |
|---|---|
| [`feedback_loop/`](feedback_loop/) | one full revolution: profile → schedule → advice → hint → rewrite → verdict |
| [`verbs/`](verbs/) | one script per verb — fuse, split, unfuse, shard, choose_implementation |
| [`workloads/`](workloads/) | authoring a spec, and the four fields that are load-bearing |
| [`k1_board/`](k1_board/) | build → deploy → profile → trace → Gantt on real hardware |
| [`solvers/`](solvers/) | the registered schedulers on one workload, side by side |

## Why these exist

Every one of them documents something that was got wrong at least once, and
says so in the file. A comment explaining a trap is worth more than a comment
explaining a happy path, because the happy path is legible from the code.

`examples/run_all.py` runs the subset that needs neither a board nor a MOSEK
licence, and the test suite runs it, so an example cannot rot quietly into a
description of code that no longer exists.
