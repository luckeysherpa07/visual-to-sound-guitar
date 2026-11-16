"""Inference wrapper for the trained fretboard pose estimator."""

from __future__ import annotations

import numpy as np


class FretboardPoseEstimator:
    def __init__(self, config: str, checkpoint: str, device: str = "cuda:0"):
        from mmpose.apis import init_model

        self.model = init_model(config, checkpoint, device=device)

    def predict(self, image: np.ndarray | str, bboxes_xyxy: np.ndarray) -> np.ndarray:
        """126-point grids for each box: (N, 126, 2) in image pixels.

        ``bboxes_xyxy``: (N, 4) tight fretboard boxes (padding is added by the
        model's pipeline) or detector boxes.
        """
        from mmpose.apis import inference_topdown

        bboxes = np.asarray(bboxes_xyxy, dtype=np.float32).reshape(-1, 4)
        results = inference_topdown(self.model, image, bboxes=bboxes, bbox_format="xyxy")
        return np.stack([r.pred_instances.keypoints[0] for r in results])
