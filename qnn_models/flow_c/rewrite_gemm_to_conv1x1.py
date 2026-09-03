"""Rewrite every rank-2 constant-weight Gemm into a 1x1 Conv over [1,K,N,1].

Finding 1 (SmolVLA, QRB5165 v66): QNN's FullyConnected kernel is far slower than
its Conv2d kernel. In DLC (NHWC) space a rank-2 [N,K] activation and a
[1,K,N,1] NCHW conv input have identical memory layout, so the wrapping
Transpose/Reshape pair should be elided by the converter rather than becoming
real ops (the failure mode Finding 1 warns about for rank-3 transformers).
"""
import onnx, sys, numpy as np
from onnx import helper, numpy_helper, TensorProto
src,dst=sys.argv[1],sys.argv[2]
m=onnx.load(src); g=m.graph
init={i.name:numpy_helper.to_array(i) for i in g.initializer}
# shapes by running ORT
import onnxruntime as ort
mm=onnx.load(src); ex={o.name for o in mm.graph.output}
for n in mm.graph.node:
    for o in n.output:
        if o and o not in ex: mm.graph.output.append(onnx.ValueInfoProto(name=o)); ex.add(o)
so=ort.SessionOptions(); so.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
s=ort.InferenceSession(mm.SerializeToString(),so,providers=['CPUExecutionProvider'])
rng=np.random.RandomState(2)
feed={i.name:rng.randn(*[d if isinstance(d,int) else 1 for d in i.shape]).astype(np.float32) for i in s.get_inputs()}
nm=[o.name for o in s.get_outputs()]; shp={a:tuple(b.shape) for a,b in zip(nm,s.run(nm,feed)) if hasattr(b,'shape')}
new=[];nrw=0
for n in g.node:
    if n.op_type=='MatMul' and n.input[1] in init and len(shp.get(n.input[0],()))==2:
        x=n.input[0]; W=init[n.input[1]]; B=np.zeros(W.shape[1],dtype=np.float32); tb=0; ta=0
    elif n.op_type=='Gemm' and len(n.input)>=3 and n.input[1] in init:
        x=n.input[0]; W=init[n.input[1]]; B=init[n.input[2]] if n.input[2] in init else None
        tb=0; ta=0
        for a in n.attribute:
            if a.name=='transB': tb=a.i
            if a.name=='transA': ta=a.i
    else:
        new.append(n); continue
    xs=shp.get(x)
    if xs is None or len(xs)!=2 or B is None or ta: new.append(n); continue
    N,K=xs
    Wm = W if tb else W.T           # [M,K]
    if Wm.shape[1]!=K: new.append(n); continue
    M=Wm.shape[0]
    t=n.name.replace('/','_')
    cw=numpy_helper.from_array(np.ascontiguousarray(Wm.reshape(M,K,1,1)),'cw'+t)
    cb=numpy_helper.from_array(np.ascontiguousarray(B.astype(np.float32)),'cb'+t)
    g.initializer.extend([cw,cb])
    sh1=numpy_helper.from_array(np.array([1,K,N,1],dtype=np.int64),'sh1'+t)
    sh2=numpy_helper.from_array(np.array([M,N],dtype=np.int64),'sh2'+t)
    g.initializer.extend([sh1,sh2])
    new += [helper.make_node('Transpose',[x],['ct0'+t],perm=[1,0],name='c_tr_in'+t),
            helper.make_node('Reshape',['ct0'+t,'sh1'+t],['ct1'+t],name='c_rs_in'+t),
            helper.make_node('Conv',['ct1'+t,'cw'+t,'cb'+t],['ct2'+t],kernel_shape=[1,1],strides=[1,1],pads=[0,0,0,0],dilations=[1,1],group=1,name='c_conv'+t),
            helper.make_node('Reshape',['ct2'+t,'sh2'+t],['ct3'+t],name='c_rs_out'+t),
            helper.make_node('Transpose',['ct3'+t],[n.output[0]],perm=[1,0],name='c_tr_out'+t)]
    nrw+=1
del g.node[:]; g.node.extend(new)
onnx.save(m,dst)
print("rewrote %d rank-2 Gemms to Conv1x1; nodes now %d"%(nrw,len(g.node)))
# validate
s0=ort.InferenceSession(src,providers=['CPUExecutionProvider']); s1=ort.InferenceSession(dst,providers=['CPUExecutionProvider'])
n0=[o.name for o in s0.get_outputs()]
o0=dict(zip(n0,s0.run(None,feed))); o1=dict(zip([o.name for o in s1.get_outputs()],s1.run(None,feed)))
for k in n0: print("  validate %-52s max_abs=%.3e"%(k,np.abs(o0[k]-o1[k]).max()))
