# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Makes the FunASR OpenVINO conversion robust across OpenVINO releases.
#
# Patch 0001 calls ov.convert_model() directly on the traced backbone/embedder
# modules. On modern OpenVINO the PyTorch frontend rejects the backbone graph
# (prim::ListConstruct with non-constant inputs). This script injects a fallback
# that exports the model to ONNX via torch.onnx.export when ov.convert_model
# fails, so the backbone can still be converted to IR (via ovc) afterwards.
#
# It performs targeted string replacements (robust to line-number shifts) on
# funasr/models/seaco_paraformer/export_meta.py after patch 0001 is applied.

import sys

TARGET = "funasr/models/seaco_paraformer/export_meta.py"

BB_OLD = """    ov_model = ov.convert_model(backbone_model, example_input=bb_input)
    save_model(ov_model, 'model_bb.xml', compress_to_fp16=False)
    print("== export bb_model IR success ==")"""

BB_NEW = """    try:
        ov_model = ov.convert_model(backbone_model, example_input=bb_input)
        save_model(ov_model, 'model_bb.xml', compress_to_fp16=False)
        print("== export bb_model IR success ==")
    except Exception as _e:
        print(f"== ov.convert_model(bb) failed ({_e}); falling back to ONNX ==")
        torch.onnx.export(
            backbone_model, (speech, speech_lengths, bias_embed), 'model_bb.onnx',
            input_names=["speech", "speech_lengths", "bias_embed"],
            output_names=["logits", "token_num", "alphas"],
            dynamic_axes={"speech": {0: "batch_size", 1: "feats_length"},
                          "speech_lengths": {0: "batch_size"},
                          "bias_embed": {0: "batch_size", 1: "num_hotwords"}},
            opset_version=14, dynamo=False)
        print("== export bb_model ONNX success ==")"""

EB_OLD = """    ov_eb_model = ov.convert_model(embedder_model, example_input=(hotword))
    save_model(ov_eb_model, 'model_eb.xml', compress_to_fp16=False)
    print("== export eb_model IR success ==")"""

EB_NEW = """    try:
        ov_eb_model = ov.convert_model(embedder_model, example_input=(hotword))
        save_model(ov_eb_model, 'model_eb.xml', compress_to_fp16=False)
        print("== export eb_model IR success ==")
    except Exception as _e:
        print(f"== ov.convert_model(eb) failed ({_e}); falling back to ONNX ==")
        torch.onnx.export(
            embedder_model, (hotword,), 'model_eb.onnx',
            input_names=["hotword"], output_names=["hw_embed"],
            dynamic_axes={"hotword": {0: "num_hotwords", 1: "hotword_len"}},
            opset_version=14, dynamo=False)
        print("== export eb_model ONNX success ==")"""


def main() -> int:
    with open(TARGET, "r", encoding="utf-8") as handle:
        text = handle.read()

    for old, new, label in ((BB_OLD, BB_NEW, "backbone"), (EB_OLD, EB_NEW, "embedder")):
        if new in text:
            print(f"[onnx-fallback] {label} already patched; skipping")
            continue
        if old not in text:
            print(f"[onnx-fallback] ERROR: could not locate {label} block in {TARGET}")
            return 1
        text = text.replace(old, new, 1)
        print(f"[onnx-fallback] patched {label} block")

    with open(TARGET, "w", encoding="utf-8") as handle:
        handle.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
