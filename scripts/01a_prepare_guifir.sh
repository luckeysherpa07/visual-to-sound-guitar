#!/usr/bin/env bash
# Download and prepare GuiFiR-Vid into data/raw/guifir/ using the official scripts
# (github.com/kmu-ee-ailab/guifir_dataset). Dataset terms: non-commercial research.
# Requires: pip install -e .[data]  (gdown, yt-dlp; only if the tar is not already in tmp/).
# If downloaded manually, put it at data/raw/guifir/tmp/GuiFiR.tar (it is kept).
#
# Result:
#   data/raw/guifir/guifir_vid/{train,val,test}/{videos,frames,annotations}/
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/data/raw/guifir"
OFFICIAL="$OUT/_official_repo"
mkdir -p "$OUT"
cd "$OUT"

[ -d "$OFFICIAL" ] || git clone --depth 1 https://github.com/kmu-ee-ailab/guifir_dataset.git "$OFFICIAL"
S="$OFFICIAL/scripts"

mkdir -p tmp
[ -f tmp/GuiFiR.tar ] || gdown --id 1Ri42XQnpwWQzjNLOPgZCR2XCx9YGrqxc -O tmp/GuiFiR.tar
[ -d tmp/GuiFiR ] || tar -xf tmp/GuiFiR.tar -C tmp

mkdir -p guifir_vid/metadata
cp -r tmp/GuiFiR/metadata/. guifir_vid/metadata/

python "$S/move_annotations.py" \
  --annotation_dir tmp/GuiFiR/annotations \
  --split_csv guifir_vid/metadata/split.csv \
  --output_dir guifir_vid

# Only needed if the archive has no clips.
if [ -z "$(ls -A tmp/GuiFiR/raw_clips 2>/dev/null)" ]; then
  python "$S/download_clips_from_metadata.py" \
    --yt_csv guifir_vid/metadata/yt_link.csv \
    --start_end_csv guifir_vid/metadata/start_end.csv \
    --output_dir tmp/GuiFiR/raw_clips \
    --source_dir tmp/GuiFiR/source_videos
fi

python "$S/split_clips.py" \
  --raw_dir tmp/GuiFiR/raw_clips \
  --split_csv guifir_vid/metadata/split.csv \
  --output_dir guifir_vid

# Extract ONLY the annotated frames (~31.6k), named like the annotations
# (<clip>/<clip>_NNN.png, 0-based index). The official extract_frames.py writes
# every frame as 000001.png (~300k PNGs, 100+ GB) which does not match the names.
cd "$ROOT"
for split in train val test; do
  python -c "
from guitar2midi.data import guifir
saved, missing = guifir.extract_annotated_frames('$split')
print('$split: saved', saved, 'frames, missing (index beyond clip end)', missing)
"
done

# Keep tmp/GuiFiR.tar (manual download); remove only the unpacked copy.
rm -rf "$OUT/tmp/GuiFiR"
echo "GuiFiR-Vid ready in $OUT/guifir_vid"
