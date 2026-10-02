#!/bin/bash

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Build-time provisioning for the FunASR OpenVINO benchmark.
#
# JB: Unlike act/pi05 (which download a prebuilt IR), FunASR has no published
# OpenVINO IR, so this script converts it from source:
#   1. Clone FunASR at the pinned commit.
#   2. Apply patch 0001 (enables the OpenVINO/ONNX export path).
#   3. Inject an ONNX fallback so the backbone converts on modern OpenVINO.
#   4. Run the conversion (downloads the paraformer models from ModelScope).
#   5. Convert the backbone ONNX -> IR with ovc.
#   6. Stage model_bb.* / model_eb.* (+ a speech_lengths tensor) into /resources.

set -euo pipefail

# FUNASR version 1.3.30
FUNASR_COMMIT="16cd165ac3946cc8c08bf845331f91fefec8e1a9"
ASSETS_DIR="${ASSETS_DIR:-$(pwd)}"
RESOURCES_DIR="${RESOURCES_DIR:-/resources}"
FEATS_LENGTH="${FEATS_LENGTH:-30}"

mkdir -p "${RESOURCES_DIR}"

echo "=== [1/6] Clone FunASR @ ${FUNASR_COMMIT} ==="
git clone https://github.com/modelscope/FunASR.git /opt/FunASR
cd /opt/FunASR
git checkout "${FUNASR_COMMIT}"

echo "=== [2/6] Apply patch 0001 (OpenVINO conversion enablement) ==="
dos2unix funasr/models/bicif_paraformer/cif_predictor.py
git apply "${ASSETS_DIR}/0001-OpenVINO-enable-convert-FunASR-model.patch"
# NOTE: patch 0002 (inference) is intentionally NOT applied here; it requires
# funasr.models.intel.* wrappers that are not part of the conversion flow.

echo "=== [3/6] Inject ONNX fallback for the backbone conversion ==="
python3 "${ASSETS_DIR}/funasr_onnx_fallback.py"

echo "=== [4/6] Install Python dependencies ==="
pip install --break-system-packages --no-cache-dir \
    torch torchaudio --index-url https://download.pytorch.org/whl/cpu
# OpenVINO is intentionally unpinned. The conversion (in patch 0001 /
# funasr_onnx_fallback.py) uses only the stable public API
# (openvino.convert_model / openvino.save_model), so it tracks the latest
# release (2024.x, 2025.x, 2026.x, ...) without code changes.
# rapidfuzz is an extras_require dep of FunASR >=1.3.30 (seaco_paraformer needs
# it) and is not pulled by `pip install -e`, so it is requested explicitly.
pip install --break-system-packages --no-cache-dir \
    -e /opt/FunASR modelscope onnx onnxscript openvino rapidfuzz

echo "=== [5/6] Convert FunASR -> IR (downloads models from ModelScope) ==="
# Export directly via model.export() instead of ov_convert_FunASR.py, which also
# calls model.generate() on a wav. generate() decodes audio through torchaudio ->
# torchcodec/ffmpeg (not needed for conversion and absent in a clean image), so
# it is skipped here. The export uses dummy tensors, not real audio.
#
# FunASR's own built-in ONNX export step runs *after* our IR/ONNX fallback has
# already written model_bb.onnx / model_eb.xml and can fail on newer torch; that
# failure is tolerated ("|| true") and the artifacts are verified explicitly below.
python3 - <<'PY' || true
from funasr import AutoModel

model = AutoModel(
    model="paraformer-zh", model_revision="v2.0.4",
    vad_model="fsmn-vad", vad_model_revision="v2.0.4",
    punc_model="ct-punc-c", punc_model_revision="v2.0.4",
    spk_model="cam++", spk_model_revision="v2.0.2",
)
try:
    model.export()
except Exception as export_error:
    # The FunASR-internal ONNX step may raise after our artifacts are emitted.
    print(f"[convert] model.export() raised after artifact emission: {export_error}")
PY

# Verify the artifacts produced by the OpenVINO/ONNX fallback in export_meta.py.
# Either model may be emitted as ONNX (when ov.convert_model rejects the graph on
# a given OpenVINO release); convert any such ONNX to IR with ovc.
for _m in model_bb model_eb; do
    if [ -f "${_m}.onnx" ] && [ ! -f "${_m}.xml" ]; then
        echo "Converting ${_m}.onnx -> ${_m}.xml with ovc"
        ovc "${_m}.onnx" --output_model "${_m}.xml"
    fi
done
test -f model_eb.xml || { echo "ERROR: model_eb.xml was not produced"; exit 1; }
test -f model_bb.xml || { echo "ERROR: model_bb.xml was not produced"; exit 1; }

echo "=== [6/6] Stage artifacts into ${RESOURCES_DIR} ==="
cp model_bb.xml model_bb.bin model_eb.xml model_eb.bin "${RESOURCES_DIR}/"
# speech_lengths is a data-dependent control input for the backbone; benchmark_app
# must be fed the correct value (== feats_length) instead of random data.
python3 - "${RESOURCES_DIR}/speech_lengths.npy" "${FEATS_LENGTH}" <<'PY'
import sys
import numpy as np
np.save(sys.argv[1], np.array([int(sys.argv[2])], dtype=np.int32))
print("wrote", sys.argv[1], "=", np.load(sys.argv[1]))
PY

ls -la "${RESOURCES_DIR}"
echo "=== FunASR IR provisioning complete ==="
