"""Replace exact erf-GELU with x*Sigmoid(1.702x) so the v66 DSP accepts the graph.

The QNN converter fuses Div/Erf/Add/Mul/Mul into a single ElementWiseNeuron whose
Param[0] (neuron type) = 1 (GELU); the v66 DSP op package rejects that value.
Sigmoid is accepted everywhere (the ViNT encoders run 65 of them on DSP).
"""
import onnx, sys, numpy as np
from onnx import helper, numpy_helper, TensorProto
src, dst = sys.argv[1], sys.argv[2]
m = onnx.load(src); g = m.graph
prod = {o: n for n in g.node for o in n.output}
cons = {}
for n in g.node:
    for i in n.input: cons.setdefault(i, []).append(n)
initv = {i.name: numpy_helper.to_array(i) for i in g.initializer}
def const_val(name):
    if name in initv: return initv[name]
    n = prod.get(name)
    if n is not None and n.op_type == 'Constant':
        for a in n.attribute:
            if a.name in ('value','t'): return numpy_helper.to_array(a.t)
    return None
drop = set(); add = []
nrew = 0
for n in list(g.node):
    if n.op_type != 'Erf': continue
    div = prod.get(n.input[0])
    if div is None or div.op_type != 'Div': continue
    x = div.input[0]
    s2 = const_val(div.input[1])
    if s2 is None or abs(float(s2) - np.sqrt(2.0)) > 1e-3: continue
    addn = cons.get(n.output[0], [])
    addn = [c for c in addn if c.op_type == 'Add']
    if len(addn) != 1: continue
    addn = addn[0]
    mul1 = [c for c in cons.get(addn.output[0], []) if c.op_type == 'Mul']
    if len(mul1) != 1: continue
    mul1 = mul1[0]
    if x not in mul1.input: continue
    mul2 = [c for c in cons.get(mul1.output[0], []) if c.op_type == 'Mul']
    if len(mul2) != 1: continue
    mul2 = mul2[0]
    half = None
    for i in mul2.input:
        if i != mul1.output[0]: half = const_val(i)
    if half is None or abs(float(half) - 0.5) > 1e-6: continue
    tag = n.name.replace('/', '_')
    kname = 'gelu_k' + tag
    g.initializer.append(numpy_helper.from_array(np.array([1.702], dtype=np.float32), kname))
    add.append(helper.make_node('Mul', [x, kname], ['gelu_s' + tag], name='gelu_scale' + tag))
    add.append(helper.make_node('Sigmoid', ['gelu_s' + tag], ['gelu_g' + tag], name='gelu_sig' + tag))
    add.append(helper.make_node('Mul', [x, 'gelu_g' + tag], [mul2.output[0]], name='gelu_out' + tag))
    drop |= {div.name, n.name, addn.name, mul1.name, mul2.name}
    nrew += 1
newnodes = [n for n in g.node if n.name not in drop] + add
# preserve topological order: re-sort
name2 = {n.name: n for n in newnodes}
outp = {o: n.name for n in newnodes for o in n.output}
seen = set(); order = []
def visit(nm, stack=()):
    if nm in seen: return
    seen.add(nm)
    for i in name2[nm].input:
        p = outp.get(i)
        if p: visit(p)
    order.append(name2[nm])
for n in newnodes: visit(n.name)
del g.node[:]; g.node.extend(order)
onnx.save(m, dst)
print("rewrote %d GELUs -> x*Sigmoid(1.702x); nodes %d" % (nrew, len(g.node)))
