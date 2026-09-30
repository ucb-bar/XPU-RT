Round 1 of the 90 Hz chain, first attempt (2026-09-11 17:51–18:41): the hard-window CP-SAT solve on
`cal_a90_r1.json` returned UNKNOWN at its 3000 s limit and wrote no table. The files here are what
`solve_stage2_hard.sh` copied under the round-1 name in that case — the solver's previous output for
the same spec (`scheduled_wh_chain90_solve_500_cpsat_profiled.json` of 17:46, the misaligned-emitter
round-2 table; identical md5) — and the logs of that attempt. Nothing here was executed on the board.
The solve script now clears the solver's output before each solve and falls back to the soft-window
CP-SAT table when the hard certificate is not found in the limit.
