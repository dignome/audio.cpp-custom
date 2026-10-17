"""Compare native student W4 weights, activation quantization and flow with upstream.

Uses local assets only. Requires the upstream student_flow.py dependencies in
the validation Python environment; these are not native runtime dependencies.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools/community_models'))
from kitten_tts2_student_weights import stage_student_decoder


def metrics(actual, expected):
    actual, expected = np.asarray(actual).reshape(-1), np.asarray(expected).reshape(-1)
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise RuntimeError('invalid native output shape or values')
    delta = actual - expected
    return {'max_abs':float(abs(delta).max()), 'mean_abs':float(abs(delta).mean()),
            'relative_l2':float(np.linalg.norm(delta)/max(np.linalg.norm(expected),1e-12))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--upstream',type=Path,required=True)
    parser.add_argument('--s3gen',type=Path,required=True)
    parser.add_argument('--probe',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--backend',choices=('cpu','cuda'),default='cpu')
    parser.add_argument('--rounding-sensitivity',action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(8)
    torch.set_grad_enabled(False)
    spec = importlib.util.spec_from_file_location('kitten_student_reference',args.upstream/'kittenml/kittentts2/student_flow.py')
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    checkpoint = args.model/'decoders/student_w4.pt'
    staged = args.output/'student.safetensors'
    stage_student_decoder(checkpoint,args.s3gen,staged)
    blob = torch.load(checkpoint,map_location='cpu',weights_only=True)
    flow = upstream.build_flow(**blob['arch'])
    upstream.convert_to_bitnet(flow,mode=blob['mode'],norm_input=blob['norm_input'],prequantised=True)
    state = {}
    for name,item in blob['quantised'].items():
        packed = item['packed']
        high,low = (packed>>4).to(torch.int16)-8,(packed&15).to(torch.int16)-8
        codes = torch.stack([high,low],1).reshape(-1)[:item['numel']].to(torch.int8)
        scale = item['scale'].reshape(-1,*([1]*(len(item['shape'])-1)))
        state[name] = codes.reshape(item['shape']).float()/scale
    state.update({name:value.float() for name,value in blob['dense'].items()})
    flow.load_state_dict(state,strict=True)
    flow.eval()
    voices_path = args.model/'cpp/student_w4/voices.json'
    voice = json.loads(voices_path.read_text(encoding='utf-8'))['Bruno']
    codes_path = Path(__file__).with_name('component_codes.json')
    codes = json.loads(codes_path.read_text())+[4299]*3
    token = torch.tensor([codes],dtype=torch.long)
    prompt = torch.tensor(voice['prompt_token'],dtype=torch.long)
    features = torch.tensor(voice['prompt_feat'],dtype=torch.float32)
    embedding = torch.tensor(voice['embedding'],dtype=torch.float32)
    noise = torch.zeros(1,80,2*(prompt.shape[1]+len(codes)))
    def reference():
        embedded = flow.input_embedding(torch.cat((prompt,token),1))
        encoded,_ = flow.encoder(embedded,torch.tensor([embedded.shape[1]]))
        mel,_,_ = upstream.meanflow_forward(flow,token,torch.tensor([len(codes)]),prompt,
            torch.tensor([prompt.shape[1]]),features,embedding,z=noise,n_steps=1)
        return encoded.numpy(),mel.numpy()
    encoded,mel = reference()
    activation = torch.linspace(-127.5,127.5,368).repeat(7,1)
    activation[0].zero_()
    activation[1,:12] = torch.tensor([.5,1.5,2.5,-.5,-1.5,-2.5,126.5,-126.5,0,127,-127,3.5])
    activation[1,12:] = 0
    activation[2] *= 1e-8
    activation[3] *= .019
    activation[4] *= 10
    activation.numpy().tofile(args.output/'activation-input.f32')
    expected_activation = upstream.activation_quant(activation).numpy()
    native = args.output/'native'
    env = {key.upper():value for key,value in os.environ.items()}
    if args.backend == 'cuda': env['NVIDIA_TF32_OVERRIDE'] = '0'
    with (args.output/'native.log').open('w') as log:
        subprocess.run([str(args.probe.resolve()),str(staged.resolve()),str(voices_path.resolve()),
            str(codes_path.resolve()),str(args.output.resolve()),str(native.resolve()),args.backend],
            env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    with (native/'packed-weights.f32').open('rb') as raw:
        for name in sorted(blob['quantised']):
            expected = state[name].numpy().reshape(-1)
            actual = np.frombuffer(raw.read(expected.nbytes),dtype=np.float32)
            if not np.array_equal(actual,expected): raise RuntimeError('student weight differs: '+name)
        if raw.read(1): raise RuntimeError('unexpected trailing student weights')
    report = {'backend':args.backend,'exact_packed_matrices':len(blob['quantised'])}
    for name,expected,tolerance in [('activation',expected_activation,1e-6),('encoder',encoded,.02),('mel',mel,.02)]:
        report[name] = metrics(np.fromfile(native/(name+'.f32'),dtype=np.float32),expected)
        if report[name]['relative_l2'] >= tolerance:
            raise RuntimeError(f'{name} exceeds relative L2 tolerance {tolerance}: {report[name]}')
    waveform = np.fromfile(native/'waveform.f32',dtype=np.float32)
    if waveform.size < 2400 or not np.isfinite(waveform).all() or np.mean(waveform**2) < 1e-8:
        raise RuntimeError('invalid student waveform')
    if args.rounding_sensitivity:
        original_quant = upstream.activation_quant
        upstream.activation_quant = lambda x,eps=1e-5: original_quant(torch.nextafter(x,torch.full_like(x,float('inf'))),eps)
        perturbed_encoder,perturbed_mel = reference()
        report['upstream_one_ulp_sensitivity'] = {'encoder':metrics(perturbed_encoder,encoded),'mel':metrics(perturbed_mel,mel)}
    (args.output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    main()
