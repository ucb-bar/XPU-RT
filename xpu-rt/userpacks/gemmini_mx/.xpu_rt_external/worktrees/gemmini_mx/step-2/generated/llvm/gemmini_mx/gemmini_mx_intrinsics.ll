; RUN: opt -passes=instcombine %s -S | FileCheck %s
; Generated smoke test for gemmini_mx

declare llvm_void_ty @llvm.gemmini_mx.dispatch()
