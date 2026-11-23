import numpy as np
import pretty_midi
import pytest

from guitar2midi.evaluation import note_metrics as M

DUR = 8.0


def notes(rows):
    return np.array(rows, dtype=float).reshape(-1, 3)


REF = notes([(0.5, 1.0, 60), (1.0, 1.5, 64), (2.0, 3.0, 67), (4.0, 4.5, 40), (6.0, 7.5, 52)])


def test_identical_is_perfect():
    m = M.evaluate(REF, REF, DUR)
    for k in ("onset_f50", "onset_f100", "onset_offset_f50", "frame_f", "chroma_sim", "onset_corr"):
        assert m[k] == pytest.approx(1.0), k


def test_silence_is_zero():
    m = M.evaluate(REF, notes([]), DUR)
    for k in ("onset_f50", "onset_f100", "frame_f", "chroma_sim", "onset_corr"):
        assert m[k] == 0.0, k


def test_onset_tolerance():
    shifted = REF.copy()
    shifted[:, :2] += 0.07  # 70 ms late: miss at 50 ms, hit at 100 ms
    m = M.evaluate(REF, shifted, DUR)
    assert m["onset_f50"] == 0.0
    assert m["onset_f100"] == pytest.approx(1.0)


def test_octave_error_keeps_chroma_but_not_f1():
    octave = REF.copy()
    octave[:, 2] += 12
    m = M.evaluate(REF, octave, DUR)
    assert m["onset_f50"] == 0.0
    assert m["chroma_sim"] == pytest.approx(1.0)
    assert m["onset_corr"] == pytest.approx(1.0)


def test_half_the_notes():
    m = M.evaluate(REF, REF[::2], DUR)  # 3 of 5 notes, all correct
    assert m["onset_p50"] == pytest.approx(1.0)
    assert m["onset_r50"] == pytest.approx(3 / 5)


def test_load_notes_roundtrip(tmp_path):
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(0)
    inst.notes = [pretty_midi.Note(100, int(p), s, e) for s, e, p in REF[::-1]]
    pm.instruments.append(inst)
    pm.write(str(tmp_path / "x.mid"))
    np.testing.assert_allclose(M.load_notes(tmp_path / "x.mid"), REF, atol=1e-3)
