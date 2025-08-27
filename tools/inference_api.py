"""Simple REST API for rotated object detection inference.

Usage:
  pwsh> python tools/inference_api.py \
            --config configs/lsknet/lsk_s_fpn_1x_dota_le90.py \
            --checkpoint ./data/pretrained/lsk_s_fpn_1x_dota_le90_20230116-99749191.pth \
            --host 0.0.0.0 --port 8000 --device cuda:0

Endpoints:
  POST /predict (multipart/form-data)
    - image: uploaded image file
    - score_thr: optional float (default 0.3)
    - patch_infer: optional bool (default false) use sliding window patch inference (for huge images)
    - return_image: optional bool (default false) return visualization image (base64 PNG)
    - sizes, steps, ratios, merge_iou_thr: optional JSON-style strings for patch inference parameters

Response JSON:
  {
    "time_ms": 42.7,
    "detections": [
        {"label": "plane", "score": 0.92, "bbox": [x_ctr, y_ctr, w, h, angle] },
        ...
    ],
    "image_base64": "..."  # only if return_image=true
  }

Notes:
  - Returned bbox format depends on model task (OBB/HBB). For oriented (OBB) heads the per-class
    ndarray usually has shape (N, 6): (xc, yc, w, h, angle, score). We separate score.
  - For HBB (horizontal) tasks shape is (N, 5): (x1, y1, x2, y2, score). We keep bbox list accordingly.
  - Patch inference parameters have safe defaults; adjust as needed for very large images.
"""
from __future__ import annotations

import argparse
import base64
import json
import time
from typing import Any, List

import mmcv
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import cv2

from mmdet.apis import init_detector, inference_detector

try:  # optional import (only needed if using patch inference)
    from mmrotate.apis import inference_detector_by_patches  # type: ignore
except Exception:  # pragma: no cover
    inference_detector_by_patches = None  # type: ignore

app = FastAPI(title="LSKNet Rotated Detection API", version="0.1.0")
MODEL = None  # global model reference

def load_model(config: str, checkpoint: str, device: str):
    global MODEL
    if MODEL is None:
        MODEL = init_detector(config, checkpoint, device=device)
    return MODEL

class Detection(BaseModel):
    label: str
    score: float
    bbox: List[float]

class PredictResponse(BaseModel):
    time_ms: float
    detections: List[Detection]
    image_base64: Optional[str] = None

def _to_base64(img: np.ndarray) -> str:
    success, buf = cv2.imencode('.png', img)
    if not success:
        raise RuntimeError('Failed to encode image for base64 output')
    return base64.b64encode(buf.tobytes()).decode('utf-8')

def _format_results(result: Any, class_names: List[str], score_thr: float) -> List[dict]:
    detections = []
    if isinstance(result, tuple):  # (det, segm)
        result = result[0]
    for cls_idx, cls_dets in enumerate(result):
        if cls_dets is None or len(cls_dets) == 0:
            continue
        for det in cls_dets:
            if det.shape[0] < 5:
                continue
            score = float(det[-1])
            if score < score_thr:
                continue
            if det.shape[0] == 6:  # obb
                bbox = det[:5].tolist()
            elif det.shape[0] == 5:  # hbb
                bbox = det[:4].tolist()
            else:
                bbox = det[:-1].tolist()
            detections.append({
                'label': class_names[cls_idx],
                'score': score,
                'bbox': [float(x) for x in bbox]
            })
    return detections

@app.post('/predict', response_model=PredictResponse)
async def predict(
    image: UploadFile = File(..., description="Image file"),
    score_thr: float = Form(0.3),
    patch_infer: bool = Form(False),
    return_image: bool = Form(False),
    sizes: str = Form('[1024]'),
    steps: str = Form('[824]'),
    ratios: str = Form('[1.0]'),
    merge_iou_thr: float = Form(0.1),
):
    if MODEL is None:
        raise HTTPException(status_code=503, detail='Model not loaded yet.')
    file_bytes = await image.read()
    try:
        img = mmcv.imfrombytes(file_bytes, flag='color')
    except Exception as e:  # pragma: no cover
        raise HTTPException(status_code=400, detail=f'Invalid image: {e}')
    if img is None:
        raise HTTPException(status_code=400, detail='Failed to decode image')

    start = time.time()
    if patch_infer:
        if inference_detector_by_patches is None:
            raise HTTPException(status_code=500, detail='Patch inference not available')
        try:
            sizes_list = json.loads(sizes)
            steps_list = json.loads(steps)
            ratios_list = json.loads(ratios)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail='sizes/steps/ratios must be JSON lists')
        result = inference_detector_by_patches(
            MODEL,
            img,
            sizes=sizes_list,
            steps=steps_list,
            ratios=ratios_list,
            merge_iou_thr=merge_iou_thr,
            bs=1,
        )
    else:
        result = inference_detector(MODEL, img)
    infer_time_ms = (time.time() - start) * 1000.0

    detections = _format_results(result, list(MODEL.CLASSES), score_thr)

    img_b64 = None
    if return_image:
        vis = MODEL.show_result(img.copy(), result, score_thr=score_thr, show=False, wait_time=0, out_file=None)
        if vis is not None:
            img_b64 = _to_base64(vis)

    return JSONResponse(content={
        'time_ms': round(infer_time_ms, 2),
        'detections': detections,
        'image_base64': img_b64,
    })

def parse_args():
    parser = argparse.ArgumentParser(description='Start FastAPI inference server')
    parser.add_argument('--config', required=True, help='Config file path')
    parser.add_argument('--checkpoint', required=True, help='Checkpoint file path')
    parser.add_argument('--device', default='cuda:0', help='Device, e.g. cuda:0 or cpu')
    parser.add_argument('--host', default='127.0.0.1', help='Host address')
    parser.add_argument('--port', type=int, default=8000, help='Port number')
    parser.add_argument('--workers', type=int, default=1, help='Uvicorn workers (processes)')
    return parser.parse_args()

def main():
    args = parse_args()
    load_model(args.config, args.checkpoint, args.device)
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, workers=args.workers, log_level='info')
    # uvicorn.run('tools.inference_api:app', host=args.host, port=args.port, workers=args.workers, log_level='info')

if __name__ == '__main__':  # pragma: no cover
    main()
