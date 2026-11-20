import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'v12'),str(ROOT/'v15')]
from rfpu_v15.neural import build
from rfpu_v15.paths import DATA
from rfpu_v12.checkpoint import construct
if __name__=='__main__':
    result=construct(build(),str(DATA/'checkpoints/rfpu-v15/bitwise-I16-W64-v1'))
    (ROOT/'v15/reports/checkpoint.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
