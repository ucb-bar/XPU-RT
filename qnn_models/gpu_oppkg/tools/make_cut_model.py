#!/usr/bin/env python3
"""Truncate a converter-generated QNN model .cpp after node N.

Used to bisect where a custom GPU op package's chain breaks: build a graph that
stops right after the op under test, with that op's output promoted to APP_READ
so qnn-net-run writes it out.  Also rewrites the op package name so the same
source can be pointed at either the stock qti.aisw kernels or ours.
"""
import re, sys

def build(src, out_path, last_node, package):
    text = open(src).read()
    # 1. package name for every node
    text = text.replace('"qti.aisw", // Package Name', f'"{package}", // Package Name')
    # 2. find the compose call list and cut after the requested node
    lines = text.split('\n')
    out, cutting = [], False
    for ln in lines:
        m = re.match(r'\s*VALIDATE\(addNode_(\S+?)\(', ln)
        if cutting and re.match(r'\s*VALIDATE\(add(Node|Tensor)_', ln):
            continue
        out.append(ln)
        if m and m.group(1).startswith(last_node):
            cutting = True
    text = '\n'.join(out)
    # 3. promote that node's output tensor to APP_READ
    #    (the tensor is declared inline in the addNode function)
    fn = re.search(r'static ModelError_t addNode_' + re.escape(last_node) +
                   r'\(QnnModel& model\)\{.*?\n\}\n', text, re.S)
    if not fn:
        raise SystemExit(f'node {last_node} not found')
    body = fn.group(0)
    new_body = body.replace('QNN_TENSOR_TYPE_NATIVE', 'QNN_TENSOR_TYPE_APP_READ')
    text = text.replace(body, new_body)
    open(out_path, 'w').write(text)
    print(f'wrote {out_path}: cut after {last_node}, package {package}')

if __name__ == '__main__':
    build(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
