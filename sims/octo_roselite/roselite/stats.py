from scipy.stats import fisher_exact
def f(a, na, b, nb, lbl):
    p = fisher_exact([[a, na-a], [b, nb-b]])[1]
    sig = "  ***" if p < 0.01 else ("  *" if p < 0.05 else "   n.s.")
    print(f"{lbl:60s} {a:3d}/{na:3d}={100*a/na:5.1f}%  vs {b:3d}/{nb:3d}={100*b/nb:5.1f}%   p={p:.2e}{sig}")
print("Fisher exact, two-sided (MEASURED counts)")
print("-" * 120)
f(41, 72, 38, 72, "PIPELINED 283.4ms  vs  stock free-running baseline")
f(15, 72, 38, 72, "PIPELINED 555ms    vs  stock free-running baseline")
f(15, 72, 41, 72, "PIPELINED 555ms    vs  PIPELINED 283.4ms")
print()
f(17, 120, 38, 72, "SERIAL 0ms (no ensembler)  vs  stock baseline (ens ON)")
f(19, 120, 17, 120, "SERIAL 283.4ms  vs  SERIAL 0ms (both no ensembler)")
f(7, 72, 17, 120, "SERIAL 555ms    vs  SERIAL 0ms (both no ensembler)")
f(4, 120, 17, 120, "SERIAL 684.8ms  vs  SERIAL 0ms (both no ensembler)")
f(4, 120, 19, 120, "SERIAL 684.8ms  vs  SERIAL 283.4ms")
print()
f(41, 72, 19, 120, "PIPELINED 283.4ms vs SERIAL 283.4ms (value of ensembler)")
