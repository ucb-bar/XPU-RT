#!/usr/bin/env python3
"""Bit-exact fp64 emulation of the quantized mlp_control graph.

This is the third opinion the GPU kernels are checked against: it implements
QNN's own quantization semantics from QnnTypes.h --

    real = (quantized + offset) * scale
    quantized = round(real / scale) - offset, clamped to the storage range

-- in double precision, from the same weight blob the model library links
(model/bin_extract/*.raw, untarred from the converter's .bin).  It exists
because the QNN *CPU* backend's int8 path for this graph is broken (it returns
the same output for every input; see README §5), so "compare against CPU" is
not available for mlp_control and something trustworthy had to take its place.
The DSP backend agrees with this emulation to within 2 LSB.

  python3 tools/emulate_mlp_int8.py                     # emulate 8 fixed inputs
  python3 tools/emulate_mlp_int8.py --compare <dumpdir> # and diff against a run
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Encodings as the converter emitted them (model/mlp_ref_net.json).
QP = {
    'obs': (0.019357390701770782, -132), 'g0': (0.055248524993658066, -116),
    'e1': (0.0340835265815258, -29),     'g2': (0.2128865122795105, -159),
    'e3': (0.08380891382694244, -12),    'g4': (0.16541166603565216, -136),
    'e5': (0.08131951838731766, -12),    'out': (0.12165010720491409, -40),
    'b0': (0.003489759285002947, -114),  'w0': (0.024783866479992867, -128),
    'b2': (0.002529808320105076, -199),  'w2': (0.006023322232067585, -152),
    'b4': (0.0017542074201628566, -218), 'w4': (0.0029929140582680702, -130),
    'b6': (0.00012916824198327959, -213), 'w6': (0.0026473214384168386, -98),
}
SHAPES = {0: (256, 16), 2: (128, 256), 4: (64, 128), 6: (4, 64)}


def _raw(name):
    return np.fromfile(os.path.join(HERE, 'model', 'bin_extract', name + '.raw'), dtype=np.uint8)


def quantize(real, key):
    scale, offset = QP[key]
    return np.clip(np.round(real / scale) - offset, 0, 255).astype(np.uint8)


def dequantize(q, key):
    scale, offset = QP[key]
    return (q.astype(np.int64) + offset) * scale


def fully_connected(q_in, in_key, layer, out_key):
    w = _raw(f'mlp_{layer}_weight_permute').reshape(SHAPES[layer]).astype(np.int64) \
        + QP[f'w{layer}'][1]
    b = _raw(f'mlp_{layer}_bias').astype(np.int64) + QP[f'b{layer}'][1]
    acc = w @ (q_in.astype(np.int64) + QP[in_key][1])          # exact integer reduction
    real = acc * (QP[in_key][0] * QP[f'w{layer}'][0]) + b * QP[f'b{layer}'][0]
    return quantize(real, out_key)


def elu(q_in, in_key, out_key):
    x = dequantize(q_in, in_key).astype(np.float64)
    return quantize(np.where(x >= 0.0, x, np.expm1(x)), out_key)


def run(q_obs):
    q = fully_connected(q_obs, 'obs', 0, 'g0')
    q = elu(q, 'g0', 'e1')
    q = fully_connected(q, 'e1', 2, 'g2')
    q = elu(q, 'g2', 'e3')
    q = fully_connected(q, 'e3', 4, 'g4')
    q = elu(q, 'g4', 'e5')
    return fully_connected(q, 'e5', 6, 'out')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', default=os.path.join(HERE, 'validation', 'inputs'))
    ap.add_argument('--compare', help='directory of Result_<i>/output_native.raw to diff against')
    ap.add_argument('--n', type=int, default=8)
    args = ap.parse_args()

    emulated, measured = [], []
    for i in range(args.n):
        q = np.fromfile(os.path.join(args.inputs, f'obs_q_{i}.raw'), dtype=np.uint8)
        emulated.append(run(q))
        if args.compare:
            measured.append(np.fromfile(
                os.path.join(args.compare, f'Result_{i}', 'output_native.raw'), dtype=np.uint8))
    emulated = np.stack(emulated)
    print('emulated int8 outputs:\n', emulated)
    if args.compare:
        measured = np.stack(measured)
        diff = np.abs(emulated.astype(int) - measured.astype(int))
        print(f'exact match: {100 * np.mean(emulated == measured):.2f}%   '
              f'max |diff|: {diff.max()} LSB   mean: {diff.mean():.3f} LSB')
        if np.all(measured == measured[0]):
            print('WARNING: the measured backend returned the same output for every input')
        return 0 if diff.max() == 0 else 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
