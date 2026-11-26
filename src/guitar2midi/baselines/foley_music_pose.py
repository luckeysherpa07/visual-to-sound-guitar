"""Body + hand keypoints in Foley Music's ``body65`` layout, from MMPose whole-body (env ``guitar2midi``).

Foley Music (Gan et al., ECCV 2020) was trained on OpenPose keypoints: BODY_25 (25 points) + 21 points per hand,
stored as ``body65`` = the 25 body points, then hand points 1..20 of the left hand (indices 25..44) and of the
right hand (45..64); the hand wrists are body points 7 (left) and 4 (right). Array (T, 3, 65, 1): x, y, confidence,
22 fps, missing points (0, 0, 0). OpenPose is not installed here, so we use MMPose's COCO-WholeBody model
(RTMPose-m + RTMDet-m person detector, ``MMPoseInferencer('wholebody')``) and map its 133 points:

- BODY_25 from COCO body (17) + feet (6): neck = mean of the shoulders, mid-hip = mean of the hips (confidence =
  the lower of the two), as OpenPose defines them.
- Hands: COCO-WholeBody hands use OpenPose's 21-point order (wrist, thumb 1-4, index 1-4, ...); left = the
  person's left hand in both.

The person with the largest box is kept (Foley Music stores one person). Coordinates are mapped to Foley Music's
scale with ``normalise``, calibrated with 33a_calibrate_foley_pose.py against their released keypoints of 4 MUSIC
acoustic-guitar validation videos (30 s each, same YouTube videos):
- scale: x' = x / W - 0.5, y' = y / H - 0.5 (the two cleanest videos fit with scale 0.99-1.00, offset -0.49..-0.50);
  median point error then 2-5 % of the neck-midhip distance on those videos (hands 2-11 %).
- timing: their frame i matches our keypoints at time (i - 2) / 22 s on all 4 videos (LAG_FRAMES = 2, ~90 ms);
  the model was trained with that lag, so our arrays are built the same way.
- missing points: OpenPose stores undetected points as (0, 0, 0); their missing share is 0-76 % depending on the
  video. MIN_SCORE = 0.7 = median over the 4 videos of the score threshold that reproduces their share (0.46-0.89).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

FPS = 22.0  # Foley Music's pose rate
SCALE = (1.0, 1.0)  # x' = x / W - 0.5, y' = y / H - 0.5 (see module docstring)
OFFSET = (-0.5, -0.5)
LAG_FRAMES = 2
MIN_SCORE = 0.7

# BODY_25 index -> COCO-WholeBody index (None = derived)
_BODY25_FROM_COCO = {0: 0, 2: 6, 3: 8, 4: 10, 5: 5, 6: 7, 7: 9, 9: 12, 10: 14, 11: 16, 12: 11, 13: 13, 14: 15,
                     15: 2, 16: 1, 17: 4, 18: 3, 19: 17, 20: 18, 21: 19, 22: 20, 23: 21, 24: 22}
_LHAND, _RHAND = 91, 112  # COCO-WholeBody hand starts (21 points each, wrist first)


def coco133_to_body65(kpts: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """(133, 2) pixel keypoints + (133,) scores -> (3, 65) [x, y, score] in pixels (no thresholding)."""
    out = np.zeros((3, 65), np.float32)
    for b, c in _BODY25_FROM_COCO.items():
        out[:2, b], out[2, b] = kpts[c], scores[c]
    for b, (i, j) in ((1, (5, 6)), (8, (11, 12))):  # neck, mid-hip
        out[:2, b] = (kpts[i] + kpts[j]) / 2
        out[2, b] = min(scores[i], scores[j])
    for start, base in ((_LHAND, 25), (_RHAND, 45)):
        out[:2, base:base + 20] = kpts[start + 1:start + 21].T
        out[2, base:base + 20] = scores[start + 1:start + 21]
    return out


class WholeBodyPose:
    def __init__(self, device: str | None = None):
        from mmpose.apis import MMPoseInferencer

        self.inferencer = MMPoseInferencer("wholebody", device=device)

    def frame(self, img: np.ndarray) -> np.ndarray:
        """BGR image -> (3, 65) in pixels for the largest person; zeros if nobody is found."""
        res = next(self.inferencer(img, return_vis=False))
        people = res["predictions"][0]
        if not people:
            return np.zeros((3, 65), np.float32)

        def area(p):
            x0, y0, x1, y1 = np.asarray(p["bbox"]).reshape(-1)[:4]
            return (x1 - x0) * (y1 - y0)

        p = max(people, key=area)
        return coco133_to_body65(np.asarray(p["keypoints"], np.float32), np.asarray(p["keypoint_scores"], np.float32))

    def video(self, path: Path, start: float = 0.0, duration: float | None = None, fps: float = FPS,
              progress: bool = False) -> tuple[np.ndarray, tuple[int, int]]:
        """Keypoints at ``fps`` (frame i at time start + i / fps, nearest video frame) -> ((T, 3, 65) px, (W, H))."""
        import cv2

        cap = cv2.VideoCapture(str(path))
        vfps = cap.get(cv2.CAP_PROP_FPS)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        end = n / vfps if duration is None else min(start + duration, n / vfps)
        wanted = [int(round((start + i / fps) * vfps)) for i in range(int(round((end - start) * fps)))]
        wanted = [min(w, n - 1) for w in wanted]
        out, idx, img = [], -1, None
        for k, w in enumerate(wanted):
            if w != idx:
                if w != idx + 1:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, w)
                ok, img = cap.read()
                if not ok:
                    break
                idx = w
            out.append(self.frame(img))
            if progress and k % 100 == 0:
                print(f"  pose {k}/{len(wanted)}", flush=True)
        cap.release()
        return np.stack(out), size


def normalise(pose_px: np.ndarray, size: tuple[int, int], scale: tuple[float, float] = SCALE,
              offset: tuple[float, float] = OFFSET, min_score: float = MIN_SCORE) -> np.ndarray:
    """(T, 3, 65) pixels -> Foley Music scale: x' = scale_x * x / W + offset_x (same for y); points with
    score < min_score -> (0, 0, 0), as OpenPose stores undetected points. Returns (T, 3, 65, 1) float32."""
    W, H = size
    p = pose_px.astype(np.float32).copy()
    p[:, 0] = scale[0] * p[:, 0] / W + offset[0]
    p[:, 1] = scale[1] * p[:, 1] / H + offset[1]
    miss = p[:, 2] < min_score
    p[:, 0][miss] = p[:, 1][miss] = p[:, 2][miss] = 0
    return p[..., None]


def clip_pose(est: "WholeBodyPose", video: Path, duration: float, start: float = 0.0) -> np.ndarray:
    """Foley-Music-ready keypoints of a clip: (T, 3, 65, 1) at 22 fps, frame i = video time start + (i - LAG) / 22
    (clamped at the clip start), normalised; T = round(duration * 22)."""
    t0 = max(start - LAG_FRAMES / FPS, 0.0)
    pose, size = est.video(video, t0, duration + LAG_FRAMES / FPS)
    lag = int(round((start - t0) * FPS))
    pad = LAG_FRAMES - lag  # frames before the clip start -> repeat the first frame
    if pad > 0:
        pose = np.concatenate([np.repeat(pose[:1], pad, axis=0), pose])
    n = int(round(duration * FPS))
    pose = pose[:n]
    if len(pose) < n:
        pose = np.concatenate([pose, np.repeat(pose[-1:], n - len(pose), axis=0)])
    return normalise(pose, size)
