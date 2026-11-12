"""Run after the exact v2.0 offline handoff has been installed in this test VM."""
import json,subprocess,time
from pathlib import Path
A=Path('/workspace/goodvision/acceptance');K=Path('/workspace/goodvision/GoodVision');S=K/'state';O=A/'results';O.mkdir(exist_ok=True)
def run(args,timeout=900,log=None):
 print('RUN',args,flush=True)
 with (O/log).open('w') if log else __import__('contextlib').nullcontext(None) as f:
  return subprocess.run(args,check=True,timeout=timeout,stdout=f,stderr=subprocess.STDOUT if f else None)
def capture(args):return subprocess.check_output(args,text=True)
def probe(module,suffix='initial'):
 run(['docker','run','--rm','--user','0','--pull','never','--network','host','-v',str(A)+':/acceptance','-v',str(S)+':/state:ro','--entrypoint','python','goodvision-main:pr75-603f20d','/acceptance/workflow_probe.py',module,'--suffix',suffix,'--frames-dir','/acceptance/frames','--output','/acceptance/results/'+module+'-'+suffix+'.json','--contour-key-file','/state/secrets/contour-key'],timeout=600,log=module+'-'+suffix+'.log')
def compose(module,*args):return ['docker','compose','-f',str(S/f'compose.{module}.json'),*args]
summary={'started_at':time.time(),'steps':{},'images':json.loads(capture(['docker','image','ls','--format','json'])) if False else []}
try:
 for version in ('2.10','2.8'):
  run(['docker','run','--rm','--pull','never','--network','none','--gpus','device=0','-v',str(A)+':/acceptance:ro','--entrypoint','python',f'goodvision-gpu-runtime:pr75-4c10721-torch{version}-cu128','/acceptance/gpu_runtime_probe.py'],log='runtime-'+version+'.json');summary['steps']['runtime-'+version]=True
 for module in ('bbox','contour','system1'):
  probe(module);summary['steps'][module]=True
 for module in ('bbox','contour','system1'):
  run(compose(module,'stop','--timeout','90'));run(compose(module,'up','-d','--no-build','--pull','never','--wait','--wait-timeout','900'));probe(module,'restart');summary['steps'][module+'-restart']=True
 # Same verified System1 image supplies the already-shipped standalone CLIP service.
 run(compose('system1','stop','--timeout','90'))
 run(['docker','run','-d','--pull','never','--name','goodvision-system1-clip-check','--gpus','device=0','--network','goodvision-production','--network-alias','gv-neural','-p','127.0.0.1:28098:8098','-e','HF_HUB_OFFLINE=1','-v',str(S/'models/neural')+':/models','goodvision-system1:pr75-603f20d','python','-m','goodvision_harness.neural_service','--host','0.0.0.0','--port','8098','--cache-dir','/models/huggingface'])
 for _ in range(90):
  try:
   import urllib.request
   with urllib.request.urlopen('http://127.0.0.1:28098/health',timeout=2) as response:assert json.load(response)['ready'];break
  except (OSError,AssertionError):time.sleep(2)
 else:raise RuntimeError('CLIP failed to become ready')
 probe('neural');summary['steps']['neural']=True
 run(['docker','restart','goodvision-system1-clip-check'])
 for _ in range(90):
  try:
   with urllib.request.urlopen('http://127.0.0.1:28098/health',timeout=2) as response:assert json.load(response)['ready'];break
  except (OSError,AssertionError):time.sleep(2)
 else:raise RuntimeError('CLIP failed after restart')
 probe('neural','restart');summary['steps']['neural-restart']=True
 run(['docker','rm','-f','goodvision-system1-clip-check']);run(compose('system1','up','-d','--no-build','--pull','never','--wait','--wait-timeout','900'))
 summary['passed']=True
except BaseException as e:
 summary.update(passed=False,error=type(e).__name__,detail=str(e));raise
finally:
 for name in capture(['docker','ps','-a','--format','{{.Names}}']).splitlines():
  if name.startswith('goodvision-'):
   with (O/(name+'-container.log')).open('w') as f:subprocess.run(['docker','logs','--tail','180',name],stdout=f,stderr=subprocess.STDOUT)
 summary['elapsed_seconds']=time.time()-summary['started_at'];(O/'acceptance-summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
