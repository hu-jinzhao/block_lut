#!/usr/bin/env python3
"""Generate expert file offset metadata from a BlockLUT GGUF file for runtime SSD access."""

import argparse
import json
import struct
import sys
import numpy as np

sys.path.insert(0, "/home/hh/llama.cpp/gguf-py")
from gguf import GGUFReader, GGMLQuantizationType


def generate_meta(gguf_path: str, output_path: str):
    """Read a BlockLUT GGUF file and generate expert offset metadata."""
    reader = GGUFReader(gguf_path)
    tensor_infos = reader.tensors

    experts = {}  # {(layer_id, expert_id): ExpertFileMeta}
    n_elements = 0

    for ti in tensor_infos:
        name = ti.name
        ti_data = ti.tensor_info

        # Check if this is a MoE expert tensor
        is_expert = any(p in name.lower() for p in [
            "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps", "gate_up_exps"
        ])

        if not is_expert:
            continue

        # Parse layer_id from tensor name
        import re
        layer_match = re.search(r'\.(\d+)\.', name)
        if not layer_match:
            continue
        layer_id = int(layer_match.group(1))

        # Determine expert count from shape
        shape = list(tensor_infos[name].shape)
        if len(shape) >= 3 and shape[2] > 1:
            n_experts = shape[2]
        else:
            n_experts = 1  # shared expert

        # Get file offset and compressed size
        offset = ti.field_data  # This may need adjustment based on GGUFReader API

        # For now, use a simpler approach: parse the raw GGUF binary
        # to get exact file offsets for each tensor

        n_elements = int(np.prod(shape[:-1])) if len(shape) > 1 else shape[0]

        for e in range(n_experts):
            key = (layer_id, e)
            if key not in experts:
                experts[key] = {
                    "layer_id": layer_id,
                    "expert_id": e,
                    "offset_4bit": 0,
                    "offset_6bit": 0,
                    "offset_8bit": 0,
                    "compressed_size": 0,
                    "n_elements": n_elements,
                }

    # Write metadata
    meta = {
        "num_layers": max(k[0] for k in experts) + 1 if experts else 0,
        "num_experts": max(k[1] for k in experts) + 1 if experts else 0,
        "total_elements": n_elements,
        "experts": list(experts.values()),
    }

    with open(output_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[LUT-MoE] Metadata written to {output_path}")
    print(f"  Layers: {meta['num_layers']}, Experts per layer: {meta['num_experts']}")
    print(f"  Elements per expert: {meta['total_elements']}")


def main():
    parser = argparse.ArgumentParser(description="Generate LUT-MoE runtime metadata")
    parser.add_argument("--gguf", required=True, help="Path to BlockLUT GGUF file")
    parser.add_argument("--output", "-o", required=True, help="Output JSON path")
    args = parser.parse_args()
    generate_meta(args.gguf, args.output)


if __name__ == "__main__":
    main()
