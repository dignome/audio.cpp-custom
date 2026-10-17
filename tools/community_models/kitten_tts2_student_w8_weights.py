"""Extract the shipped frozen student_w8 export without requantization.

Its graph is identical to the companion W4 export. Match W4 constants against
the packed checkpoint to recover tensor names, then take W8 constants at those
same graph locations. Constant-folded one-step time projections remain folded.
TorchScript is used offline only.
"""
from collections import defaultdict
import hashlib
from pathlib import Path
import re
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file


def _signature(tensor):
    value = tensor.detach().cpu().contiguous()
    return tuple(value.shape), str(value.dtype), hashlib.sha256(value.numpy().tobytes()).digest()


def _recover_int8(weight):
    rows = weight.detach().cpu().float().contiguous().numpy().reshape(weight.shape[0], -1)
    maximum = np.abs(rows).max(axis=1)
    if not np.isfinite(rows).all() or (maximum <= 0).any():
        raise ValueError("unsupported W8 weight row")
    found = np.zeros(len(rows), dtype=bool)
    scales = np.zeros(len(rows), dtype=np.float32)
    codes = np.zeros(rows.shape, dtype=np.int8)
    # Inverting rounded F32 division can move the candidate scale by one ULP.
    # Accept only bit-exact reconstruction of every exported F32 value.
    for qmax in (127, 128):
        guess = np.float32(qmax) / maximum
        for ulps in (0, -1, 1, -2, 2, -3, 3, -4, 4):
            candidate = (guess.view(np.int32) + ulps).view(np.float32)
            q = np.rint(rows * candidate[:, None])
            restored = q.astype(np.float32) / candidate[:, None]
            valid = (q >= -128).all(axis=1) & (q <= 127).all(axis=1)
            exact = (restored.view(np.uint32) == rows.view(np.uint32)).all(axis=1)
            take = valid & exact & ~found
            scales[take] = candidate[take]
            codes[take] = q[take].astype(np.int8)
            found |= take
            if found.all():
                return torch.from_numpy(codes.reshape(weight.shape).copy()), torch.from_numpy(scales)
    raise ValueError("W8 export is not exactly recoverable as per-channel INT8")


def read_frozen_student_w8(model: Path):
    checkpoint = torch.load(model / "decoders/student_w4.pt", map_location="cpu", weights_only=True)
    expected = {'encoder_output_size': 368, 'encoder_linear_units': 1472,
                'encoder_num_blocks': 6, 'encoder_num_up_blocks': 4,
                'encoder_attention_heads': 8, 'estimator_channels': 176,
                'estimator_n_blocks': 4, 'estimator_num_mid_blocks': 12,
                'estimator_num_heads': 8, 'estimator_attention_head_dim': 44}
    if checkpoint.get("arch") != expected or checkpoint.get("bits") != 4:
        raise ValueError("unsupported student architecture")
    if checkpoint.get("mode") != "w4" or checkpoint.get("norm_input") is not False:
        raise ValueError("unsupported companion W4 checkpoint")
    w4 = torch.jit.load(str(model / "cpp/student_w4/decoder.pt"), map_location="cpu").eval()
    w8 = torch.jit.load(str(model / "cpp/student_w8/decoder.pt"), map_location="cpu").eval()
    code4, constants4 = w4.code_with_constants
    code8, constants8 = w8.code_with_constants
    if code4 != code8:
        raise ValueError("W4/W8 frozen graphs differ; extraction requires review")
    c4, c8 = constants4.const_mapping, constants8.const_mapping
    if set(c4) != set(c8) or any(c4[k].shape != c8[k].shape or c4[k].dtype != c8[k].dtype for k in c4):
        raise ValueError("W4/W8 constant layouts differ")
    lookup = defaultdict(list)
    for key, value in c4.items():
        lookup[_signature(value)].append(key)
    original = dict(checkpoint["dense"])
    for name, item in checkpoint["quantised"].items():
        packed = item["packed"].flatten().to(torch.int16)
        codes = torch.stack((packed >> 4, packed & 15), dim=1).flatten()[:item["numel"]] - 8
        shape = item["shape"]
        original[name] = codes.reshape(shape).float() / item["scale"].reshape(-1, *([1] * (len(shape)-1)))
    mapped, missing = {}, set()
    for name, value in original.items():
        keys = lookup[_signature(value.float())]
        if not keys:
            missing.add(name)
        elif len({_signature(c8[k]) for k in keys}) != 1:
            raise ValueError("ambiguous frozen tensor: " + name)
        else:
            mapped[name] = c8[keys[0]].clone()
    prefix = "decoder.estimator."
    blocks = [prefix + "down_blocks.0.0"] + [prefix + f"mid_blocks.{i}.0" for i in range(12)] + [prefix + "up_blocks.0.0"]
    time_linears = [prefix + "time_mlp.linear_1", prefix + "time_mlp.linear_2", prefix + "time_embed_mixer"] + [b + ".mlp.1" for b in blocks]
    expected_missing = {name + ".weight" for name in time_linears}
    expected_missing |= {name + ".bias" for name in time_linears if not name.endswith("time_embed_mixer")}
    if missing != expected_missing:
        raise ValueError("unexpected constant folding in student W8 export")
    folded_keys = re.findall(r"torch.add_\(h\d*, CONSTANTS\.(c\d+)\)", code8)
    if len(folded_keys) != len(blocks) or any(tuple(c8[k].shape) != (1, 176, 1) for k in folded_keys):
        raise ValueError("unsupported one-step time conditioning")
    folded = {}
    for name in time_linears:
        shape = original[name + ".weight"].shape
        folded[name + ".weight"] = list(shape)
        if name + ".bias" in expected_missing:
            mapped[name + ".bias"] = torch.zeros(shape[0], dtype=torch.float32)
    for block, key in zip(blocks, folded_keys):
        mapped[block + ".mlp.1.bias"] = c8[key].flatten().clone()
    quantised = {}
    for name in checkpoint["quantised"]:
        if name not in mapped:
            continue
        codes, scale = _recover_int8(mapped[name])
        quantised[name] = dict(codes=codes, scale=scale, shape=list(codes.shape))
    if len(quantised) != 457 or len(folded) != 17:
        raise ValueError("unexpected frozen student projection counts")
    return dict(arch=checkpoint["arch"], quantised=quantised, folded=folded,
                dense={k: v for k, v in mapped.items() if k not in quantised})


def stage_student_w8(model: Path, teacher: Path, output: Path):
    blob = read_frozen_student_w8(model)
    tensors = {}
    with safe_open(teacher, framework="pt") as handle:
        for name in handle.keys():
            if not name.startswith("flow."):
                tensors[name] = handle.get_tensor(name)
    for name, value in blob["dense"].items():
        tensors["flow." + name] = value.float().contiguous()
    for name, item in blob["quantised"].items():
        prefix = "flow." + name
        tensors[prefix + ".packed8"] = item["codes"].flatten().contiguous()
        tensors[prefix + ".scale"] = item["scale"].contiguous()
        tensors[prefix + ".shape"] = torch.tensor(item["shape"], dtype=torch.int64)
    for name, shape in blob["folded"].items():
        tensors["flow." + name + ".folded"] = torch.tensor(shape, dtype=torch.int64)
    save_file(tensors, output, metadata={"kitten_tts2.student": "w8-frozen-one-step-v1"})
    return dict(decoder="student_w8", packed_matrices=457, folded_time_projections=17,
                packed_weights_exact=True)
