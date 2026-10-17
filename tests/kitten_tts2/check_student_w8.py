"""Compare the native W8 decoder directly with the shipped frozen graph."""
from pathlib import Path
import hashlib, json, os, subprocess, sys
import numpy as np
import torch
import argparse
from safetensors import safe_open
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--model',type=Path,required=True)
parser.add_argument('--s3gen',type=Path,required=True)
parser.add_argument('--probe',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--staged',type=Path,help='Reuse a previously staged W8 tensor source.')
args=parser.parse_args()
HERE=args.output.resolve();HERE.mkdir(parents=True,exist_ok=True)
REPO=Path(__file__).resolve().parents[2]
MODEL=args.model.resolve()
torch.set_num_threads(8);torch.set_grad_enabled(False)
sys.path.insert(0,str(REPO/'tools/community_models'))
from kitten_tts2_student_w8_weights import read_frozen_student_w8, stage_student_w8
blob=read_frozen_student_w8(MODEL)
m=torch.jit.load(str(MODEL/'cpp/student_w8/decoder.pt'),map_location='cpu').eval()
code,constants=m.code_with_constants
# First HiFT f0 convolution consumes the completed flow mel.
signature=lambda x: hashlib.sha256(x.contiguous().numpy().tobytes()).digest()
with safe_open(args.s3gen,framework='pt') as teacher:
    wanted=signature(teacher.get_tensor('mel2wav.f0_predictor.condnet.0.bias'))
g=m.graph.copy()
matches=[]
for n in g.nodes():
    if n.kind()=='aten::_convolution':
        w=list(n.inputs())[2].toIValue()
        if isinstance(w,torch.Tensor) and signature(w)==wanted: matches.append(n)
assert len(matches)==1
mel_value=list(matches[0].inputs())[0]
g.eraseOutput(0);g.registerOutput(mel_value)
noise=next(n for n in g.nodes() if n.kind()=='aten::randn_like')
zero=g.create('aten::zeros_like',list(noise.inputs()),1)
zero.output().setType(noise.output().type());zero.insertBefore(noise)
noise.output().replaceAllUsesWith(zero.output())
torch._C._jit_pass_dce(g)
frozen_mel=torch._C._create_function_from_graph('frozen_student_w8_mel',g)
voices=MODEL/'cpp/student_w8/voices.json'
voice=json.loads(voices.read_text())['Bruno']
codes_path=REPO/'tests/kitten_tts2/component_codes.json'
tokens=torch.tensor([json.loads(codes_path.read_text())+[4299]*3])
prompt=torch.tensor(voice['prompt_token'])
features=torch.tensor(voice['prompt_feat'],dtype=torch.float32)
embedding=torch.tensor(voice['embedding'],dtype=torch.float32)
expected=frozen_mel(m,tokens,prompt,features,embedding).numpy()
expected.tofile(HERE/'expected-mel.f32')
activation=torch.linspace(-127.5,127.5,368).repeat(7,1)
activation[0].zero_();activation[2]*=1e-8
activation.numpy().tofile(HERE/'activation-input.f32')
report={}
env=os.environ.copy()
staged=args.staged or HERE/'student.safetensors'
if not args.staged:stage_student_w8(MODEL,args.s3gen,staged)
for mode in ['f32','int8']:
    env['KITTEN_TTS2_STUDENT_LINEAR']=mode
    out=HERE/('component-'+mode)
    with (HERE/('component-'+mode+'.log')).open('w') as log:
        subprocess.run([str(args.probe.resolve()),
            str(staged.resolve()),str(voices),str(codes_path),str(HERE),str(out),'cpu'],
            env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    with (out/'packed-weights.f32').open('rb') as raw:
        for name in sorted(blob['quantised']):
            item=blob['quantised'][name]
            weight=(item['codes'].float()/item['scale'].reshape(-1,*([1]*(len(item['shape'])-1)))).numpy().flatten()
            actual=np.frombuffer(raw.read(weight.nbytes),np.float32)
            assert np.array_equal(actual.view(np.uint32),weight.view(np.uint32)),name
        assert not raw.read(1)
    actual=np.fromfile(out/'mel.f32',np.float32)
    assert actual.size==expected.size and np.isfinite(actual).all()
    waveform=np.fromfile(out/'waveform.f32',np.float32)
    assert waveform.size>2400 and np.isfinite(waveform).all() and np.mean(waveform**2)>1e-8
    delta=actual-expected.flatten()
    metrics=dict(relative_l2=float(np.linalg.norm(delta)/np.linalg.norm(expected)),max_abs=float(np.abs(delta).max()),
                 exact_weight_matrices=len(blob['quantised']))
    report[mode]=metrics
    (HERE/'component-report.json').write_text(json.dumps(report,indent=2))
    print(mode,metrics,flush=True)
    assert metrics['relative_l2']<.02
print('Frozen W8 component parity passed.')
