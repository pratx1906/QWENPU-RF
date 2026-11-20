"""Freeze history and establish resource feasibility before emitting a model."""
import sys,json,hashlib,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'v12'),str(ROOT/'v15')]
from rfpu_v15.neural import build
def main():
    from chipformer.q35.tiny import pick_layout
    files={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for folder in ['rfpu','tests','docs','reports','configs','v11','v12','v13','v14'] for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts}
    (ROOT/'v15/reports/history-freeze.json').write_text(json.dumps({'files':files},indent=2),encoding='utf-8')
    c=build();layout=pick_layout(len(c.dims))
    result={'stats':c.stats(),'hidden':layout.hidden,'GPU':subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.free,driver_version','--format=csv'],text=True)}
    (ROOT/'v15/reports/architecture-preflight.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
