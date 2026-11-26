"""Torch-2 compatibility for Foley Music (written for PyTorch 1.4); applied by sitecustomize.py at start-up when
this folder is on PYTHONPATH, or by importing this module (guitar2midi.baselines.foley_music).

torch >= 2.1 ``nn.Transformer.forward`` passes ``is_causal`` / ``tgt_is_causal`` / ``memory_is_causal`` to the
encoder and decoder; Foley Music's custom encoder/decoder (``music_transformer_dev``) have the torch-1.x signatures
and fail. We restore the torch-1.x ``nn.Transformer.forward`` (same computation: encoder, then decoder, with the
same masks). Also restores ``np.int`` for pretty_midi 0.2.9 (see the end). Nothing else is changed; the Foley Music
repo itself stays unmodified.
"""

try:
    from torch import nn

    def _forward_torch1(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None, src_key_padding_mask=None,
                        tgt_key_padding_mask=None, memory_key_padding_mask=None, **_ignored):
        if src.size(1) != tgt.size(1):
            raise RuntimeError("the batch number of src and tgt must be equal")
        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        return self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                            tgt_key_padding_mask=tgt_key_padding_mask,
                            memory_key_padding_mask=memory_key_padding_mask)

    nn.Transformer.forward = _forward_torch1
except ImportError:  # torch not installed in this interpreter
    pass

# pretty_midi 0.2.9 (pinned: 0.2.10+ rejects Note(..., end=None), used by Foley Music) still calls ``np.int``,
# removed in numpy 1.24; only MIDI reading of URMP hits it. Restore the alias (= builtin int, same behaviour).
import numpy as _np

if not hasattr(_np, "int"):
    _np.int = int
