# MMPose dataset metainfo for the 126-point fretboard grid.
# Must match src/guitar2midi/preprocessing/fretboard_grid.py (checked in tests).
# idx = string * 21 + fret; string 0 = high E, fret 0 = nut.
# swap='' -> identity flip: labels are physical, a mirrored guitar keeps them.
# joint_weights: corners 1/4, edge points 1/46, internal 1/76 (GuiFiR Sec. IV-A.2).

_S, _F = 6, 21


def _kind(s, f):
    on_s = s in (0, _S - 1)
    on_f = f in (0, _F - 1)
    if on_s and on_f:
        return 'corner'
    return 'edge' if (on_s or on_f) else 'internal'


def _color(s, f):
    if f == 0:
        return [255, 0, 0]
    return [0, 200, 0] if s in (0, _S - 1) else [0, 120, 255]


_W = dict(corner=1.0 / 4, edge=1.0 / 46, internal=1.0 / 76)

dataset_info = dict(
    dataset_name='fretboard126',
    paper_info=dict(
        author='An, Chung, Park, Lee',
        title='GuiFiR: Weakly-Supervised Learning of Hand Pose Estimation '
        'for Guitar Finger Reading',
        container='IEEE Access',
        year='2026',
        homepage='https://github.com/kmu-ee-ailab/guifir_dataset',
    ),
    keypoint_info={
        s * _F + f: dict(
            name=f's{s + 1}_f{f:02d}',
            id=s * _F + f,
            color=_color(s, f),
            type='',
            swap='')
        for s in range(_S) for f in range(_F)
    },
    skeleton_info={
        i: dict(link=link, id=i, color=[0, 200, 0])
        for i, link in enumerate(
            [(f's{s + 1}_f{f:02d}', f's{s + 1}_f{f + 1:02d}')
             for s in range(_S) for f in range(_F - 1)] +
            [(f's{s + 1}_f{f:02d}', f's{s + 2}_f{f:02d}')
             for s in range(_S - 1) for f in range(_F)])
    },
    joint_weights=[_W[_kind(s, f)] for s in range(_S) for f in range(_F)],
    # Only used by COCO OKS metrics (not used here); uniform placeholder.
    sigmas=[0.025] * (_S * _F),
)
