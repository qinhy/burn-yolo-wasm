"""Compare diagnostic GPU head with PyTorch on the EXACT GPU-preprocessed input."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from ultralytics import YOLO

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('report', nargs='?', default='model/video-validation.json')
parser.add_argument('--weights', default='yolo11n.pt')
args = parser.parse_args()
report = json.loads(Path(args.report).read_text())
torch.set_num_threads(4)
model = YOLO(args.weights).model.float().eval()
# Match ONNX export's fused Conv+BN and decoding configuration.
model.fuse(verbose=False)
with torch.no_grad():
    pixels = torch.tensor(report['input'], dtype=torch.float32).reshape(1, 3, 320, 320)
    prediction = model(pixels)
    if isinstance(prediction, (tuple, list)):
        prediction = prediction[0]
    reference = prediction.detach().cpu().numpy()
actual = np.asarray(report['head'], dtype=np.float32).reshape(report['headShape'])
error = np.abs(reference - actual)
# Box coordinates and low-confidence logits have different useful tolerances.
box_error = float(error[:, :4].max())
score_error = float(error[:, 4:].max())
passed = bool(np.allclose(reference[:, :4], actual[:, :4], atol=0.05, rtol=0.001)
              and np.allclose(reference[:, 4:], actual[:, 4:], atol=0.002, rtol=0.01))
summary = dict(passed=passed, max_box_coordinate_error=box_error,
               max_class_score_error=score_error, gpu_nms_matches_cpu=report['nmsMatchesCpu'])
print(json.dumps(summary, indent=2))
if not passed:
    raise SystemExit('GPU/Burn head differs from PyTorch beyond the declared tolerance')
