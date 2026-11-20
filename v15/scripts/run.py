"""Independent append-only experiment records; never refresh historical ledgers."""
import sys,json,time,subprocess,hashlib,zipfile
from pathlib import Path
from datetime import datetime,timezone,timedelta
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'v15'))
from rfpu_v15.textio import utf8_stdout
utf8_stdout()  # the log echo below must not crash on model text (Amendment 12 P12.2)

def refresh():
    rows=[]
    for p in sorted((ROOT/'v15/results/raw').glob('experiment-*.json')):
        x=json.loads(p.read_text());rows.append(f'| {x["name"]} | {x["status"]} | [log](results/raw/{x["log"]}) | [record](results/raw/{p.name}) |')
    summary=ROOT/'v15/reports/RESULTS_SUMMARY.md'
    intro=summary.read_text(encoding='utf-8') if summary.exists() else 'Research in progress; gate status requires measured evidence.'
    (ROOT/'v15/results.md').write_text('# RFPU-VX v1.5 results\n\nUpdated '+datetime.now(timezone(timedelta(hours=5,minutes=30))).isoformat()+'\n\n'+intro+'\n\n[Gate ledger](reports/gates.json) · [Protocol](research/PROTOCOL.md) · [Paper draft](paper/PAPER_DRAFT.md)\n\n| Experiment | Status | Full log | Record |\n|---|---|---|---|\n'+'\n'.join(rows)+'\n',encoding='utf-8')

def main():
    name=sys.argv[1];cmd=sys.argv[2:];cmd=cmd[1:] if cmd[:1]==['--'] else cmd
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');folder=ROOT/'v15/results/raw'
    stem='experiment-'+stamp+'-'+name;record=folder/(stem+'.json');log=folder/(stem+'.log')
    source={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'v15').rglob('*')
            if p.is_file() and p.suffix in ('.py','.md','.txt','.json','.bib','.tex','.s') and not any(part in ('__pycache__','results','reports','artifacts','release') for part in p.relative_to(ROOT/'v15').parts)}
    x={'name':name,'status':'running','command':cmd,'started_utc':stamp,'log':log.name,'source_sha256':source}
    snapshot=folder/(stem+'.source.zip')
    with zipfile.ZipFile(snapshot,'w',zipfile.ZIP_DEFLATED) as z:
        for relative in source:z.write(ROOT/relative,relative)
        z.writestr('SOURCE_SHA256.json',json.dumps(source,indent=2))
    x['source_archive']=snapshot.name;x['source_archive_sha256']=hashlib.sha256(snapshot.read_bytes()).hexdigest()
    def save():record.write_text(json.dumps(x,indent=2));refresh()
    save();start=time.perf_counter()
    try:
        with log.open('w',encoding='utf-8') as f:
            child=subprocess.Popen(cmd,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
            for line in child.stdout:f.write(line);f.flush();print(line,end='',flush=True)
            x['exit_code']=child.wait()
        x['status']='passed' if x['exit_code']==0 else 'failed'
    except BaseException as e:x.update(status='failed',error=repr(e));raise
    finally:x['seconds']=time.perf_counter()-start;save()
    return x['exit_code']

if __name__=='__main__':raise SystemExit(main())
