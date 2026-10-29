#!/usr/bin/env python3
"""Export Ultralytics detection weights, convert ONNX to Burn, ZIP Burnpack."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "model" / "yolo11n.onnx"
GENERATED = ROOT / "crates" / "inference" / "src" / "generated"
ZIP = ROOT / "crates" / "inference" / "src" / "model.zip"
META = ROOT / "src" / "model-meta.json"


def check_onnx(path: Path, size: int) -> tuple[int, list[int]]:
    import onnx

    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    assert len(model.graph.input) == 1, "Expected one image input"
    assert len(model.graph.output) == 1, "Expected one prediction output; disable embedded NMS"
    dims = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
    assert dims == [1, 3, size, size], f"Unexpected input shape: {dims}"
    out = [d.dim_value for d in model.graph.output[0].type.tensor_type.shape.dim]
    assert len(out) == 3 and out[0] == 1, f"Unexpected output shape: {out}"
    assert out[1] > 4 and out[2] > 0, f"Expected [1,4+classes,N], got {out}"
    return out[1] - 4, out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", default="yolo11n.pt", help="Ultralytics detection weights")
    p.add_argument("--size", type=int, default=320, help="Static square size; keep 320 for Rust demo")
    p.add_argument("--skip-export", action="store_true", help="Use model/yolo11n.onnx already present")
    args = p.parse_args()
    if args.size != 320:
        p.error("This repository fixes INPUT_SIZE=320. Update Rust + geometry and rebuild before changing it.")

    MODEL.parent.mkdir(parents=True, exist_ok=True)
    GENERATED.mkdir(parents=True, exist_ok=True)

    names = None
    if not args.skip_export:
        from ultralytics import YOLO

        net = YOLO(args.weights)
        if net.task != "detect":
            raise RuntimeError(f"Expected a detection model, got {net.task!r}")
        names = net.names
        exported = Path(net.export(
            format="onnx",
            imgsz=args.size,
            batch=1,
            dynamic=False,
            nms=False,
            opset=17,
            simplify=False,
            half=False,
        )).resolve()
        if exported != MODEL.resolve():
            shutil.copyfile(exported, MODEL)
        print(f"ONNX export: {MODEL} ({MODEL.stat().st_size / 1e6:.2f} MB)")
    if not MODEL.is_file():
        raise FileNotFoundError(f"Missing {MODEL}; remove --skip-export")

    num_classes, output_shape = check_onnx(MODEL, args.size)
    print(f"Verified ONNX output: {output_shape}, classes={num_classes}")

    if shutil.which("onnx2burn") is None:
        raise RuntimeError("onnx2burn unavailable; cargo install burn-onnx --version 0.22.0")
    subprocess.run(["onnx2burn", str(MODEL), str(GENERATED), "--no-development"], check=True, cwd=ROOT)

    rs = GENERATED / "yolo11n.rs"
    bpk = GENERATED / "yolo11n.bpk"
    if not rs.is_file() or not bpk.is_file():
        found = ", ".join(p.name for p in GENERATED.iterdir())
        raise RuntimeError(f"Expected generated yolo11n.rs and yolo11n.bpk. Found: {found}")
    with ZipFile(ZIP, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        archive.write(bpk, arcname="model.bpk")
    print(f"Burn model: {rs}")
    print(f"Burnpack: {bpk.stat().st_size:,} bytes; compressed ZIP: {ZIP.stat().st_size:,} bytes")

    if names is not None:
        classes = [names[i] for i in range(num_classes)]
        META.write_text(json.dumps({"names": classes, "size": args.size, "model": args.weights}, indent=2) + "\n")
        print(f"Updated metadata: {META}")
    else:
        current = json.loads(META.read_text())
        if len(current.get("names", [])) != num_classes:
            print("WARNING: metadata class count differs; regenerate src/model-meta.json", file=sys.stderr)
    print("Ready: npm install && npm run dev")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, FileNotFoundError, AssertionError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise SystemExit(1)
