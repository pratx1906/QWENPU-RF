import sys,json,random
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'v12'),str(ROOT/'v15')]
from rfpu_v15.runtime import Engine
def signed(v):return (v+2**63)%2**64-2**63
def main():
    import torch
    torch.set_num_threads(8);e=Engine(batch=32);rng=random.Random(1501);rows=[]
    pairs=[(a,b) for a in [-32768,-32767,-129,-1,0,1,127,32767] for b in [-32768,-255,-1,0,1,255,32767]]+[(rng.randrange(-32768,32768),rng.randrange(-32768,32768)) for _ in range(200)]
    x=[a for a,b in pairs];y=[b for a,b in pairs]
    for op in (0,1,2,4,32,33,39,46,47,62):
        actual=e.op(op,x,y).cpu().tolist()
        expected=[signed(a+b) if op==0 else signed(a-b) if op==1 else a*b if op==2 else int(a>=b) if op==4 else signed(a+(1<<(op-33)) if op>32 else a)>>(op-32) for a,b in pairs]
        assert actual==expected,(op,next((i,a,b) for i,(a,b) in enumerate(zip(actual,expected)) if a!=b))
        rows.append({'opcode':op,'elements':len(pairs),'exact':True});print('NATIVE EXACT',op,len(pairs),flush=True)
    report={'status':'passed','rows':rows,'loading_including_transfer_seconds':e.p.loading_seconds,'transfer_seconds':e.p.transfer_seconds,'capture_seconds':e.capture_seconds,'telemetry':e.telemetry()}
    (ROOT/'v15/reports/native-validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
if __name__=='__main__':main()
