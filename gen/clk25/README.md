# gen/clk25/ — 25 MHz clock-invariance control

The same measured cycle counts as `gen/`, converted at a 25 MHz clock instead of the core clock. Used by
`xpu-rt/profile_loader.py` and `xpu-rt/tests/test_gen_root.py` to show that schedules depend on cycles,
not on the clock they are converted at.
