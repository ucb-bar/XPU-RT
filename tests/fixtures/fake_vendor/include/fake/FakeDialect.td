include "mlir/IR/OpBase.td"

def FakeDialect : Dialect {
  let name = "fake";
  let cppNamespace = "::fake";
}

class Fake_Op<string mnemonic, list<Trait> traits = []>
    : Op<FakeDialect, mnemonic, traits>;

def Fake_MatmulOp : Fake_Op<"matmul"> {
  let summary = "Tiled matmul for the fake vendor";
  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor:$out);
}

def Fake_SoftmaxOp : Fake_Op<"softmax"> {
  let summary = "Row-wise softmax";
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$out);
}
