#!/usr/bin/env python3
"""Validate the ONNX contract required by the demo and optionally run a zero-input smoke test."""
from pathlib import Path
import sys
from prepare_model import check_onnx

if __name__ == '__main__':
    path = Path(sys.argv[1] if len(sys.argv) > 1 else 'model/yolo11n.onnx')
    classes, dims = check_onnx(path, 320)
    print('Valid ONNX model:', path, 'classes:', classes, 'output:', dims)
    try:
        import numpy as np
        import onnxruntime as ort
        session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
        image = np.zeros((1, 3, 320, 320), dtype=np.float32)
        y = session.run(None, {session.get_inputs()[0].name: image})[0]
        assert y.shape == tuple(dims), (y.shape, dims)
        assert np.isfinite(y).all()
        print('ONNX Runtime zero-input check passed:', y.shape)
    except ImportError:
        print('onnxruntime/numpy not installed; skipping execution check')
