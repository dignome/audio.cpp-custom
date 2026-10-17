"""Student W4 reader preserving upstream packed values and scale divisors."""
import math
import torch
from pathlib import Path
from safetensors import safe_open

def stage_student_decoder(checkpoint: Path, teacher: Path, output: Path):
    """Retain packed signed nibbles and original F32 scale divisors on disk.

    The native Kitten runtime uses INT8 linear weights on CPU; convolutions
    and the F32 reference path expand through the Kitten TensorSource.
    This preserves the upstream dequantization exactly instead of requantizing
    the student's per-output-channel F32 scales into GGML FP16 block scales.
    """
    from safetensors.torch import save_file
    blob = torch.load(checkpoint, map_location='cpu', weights_only=True)
    expected = {'encoder_output_size': 368, 'encoder_linear_units': 1472,
                'encoder_num_blocks': 6, 'encoder_num_up_blocks': 4,
                'encoder_attention_heads': 8, 'estimator_channels': 176,
                'estimator_n_blocks': 4, 'estimator_num_mid_blocks': 12,
                'estimator_num_heads': 8, 'estimator_attention_head_dim': 44}
    if blob.get('arch') != expected or blob.get('mode') != 'w4' or blob.get('bits') != 4 or blob.get('norm_input') is not False:
        raise ValueError('unsupported student architecture or activation quantization policy')
    if len(blob['quantised']) != 474:
        raise ValueError('student_w4 requires 474 packed projections')
    tensors = {}
    with safe_open(teacher, framework='pt') as handle:
        for name in handle.keys():
            if not name.startswith('flow.'):
                tensors[name] = handle.get_tensor(name)
    for name, value in blob['dense'].items():
        if name in blob['quantised'] or not torch.is_floating_point(value) or not torch.isfinite(value).all():
            raise ValueError(f'invalid student dense tensor: {name}')
        tensors['flow.' + name] = value.float().contiguous()
    for name, item in blob['quantised'].items():
        shape = item['shape']
        count = math.prod(shape)
        packed, scale = item['packed'], item['scale']
        if (len(shape) not in (2, 3) or any(dim <= 0 for dim in shape) or count != item['numel'] or
                packed.dtype != torch.uint8 or packed.numel() != (count + 1) // 2 or
                list(scale.shape) != [shape[0]] or not torch.isfinite(scale).all() or (scale <= 0).any()):
            raise ValueError(f'invalid student packed tensor: {name}')
        prefix = 'flow.' + name
        tensors[prefix + '.packed4'] = packed.flatten().view(torch.int8)
        tensors[prefix + '.scale'] = scale.float().contiguous()
        tensors[prefix + '.shape'] = torch.tensor(shape, dtype=torch.int64)
    save_file(tensors, output, metadata={'kitten_tts2.student': 'w4-v1'})
    return {'decoder': 'student_w4', 'architecture': expected,
            'packed_matrices': len(blob['quantised']), 'packed_weights_exact': True}
