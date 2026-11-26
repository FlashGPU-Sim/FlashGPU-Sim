#!/usr/bin/env python3
import json,pathlib,re,sys
p=pathlib.Path(sys.argv[1]);stem=sys.argv[2] if len(sys.argv)>2 else 'probe';s=(p/'artifacts'/(stem+'.sass')).read_text();report={}
for part in s.split('Function : ')[1:]:
 name=part.splitlines()[0].strip();bg=int(re.search(r'ELi(\d+)EEv',name)[1])
 marks=list(re.finditer(r'/\*([0-9a-f]+)\*/\s+CS2R[^;]*SR_CLOCKLO[^;]*;',part));assert len(marks)==4,(name,len(marks))
 last_clock=int(marks[-1][1],16)
 stores=[int(m[1],16) for m in re.finditer(r'/\*([0-9a-f]+)\*/\s+STG[^;]*;',part)]
 assert stores and min(stores)>last_clock,(name,'report write can overlap the other worker')
 tail=part[marks[-1].end():]
 before_store=tail[:re.search(r'\bSTG',tail).start()]
 assert 'BAR.SYNC' in before_store and re.search(r'\bLDS\s',before_store),(name,'missing joined report gate')
 intervals=[]
 for i in (0,2):
  lo,hi=int(marks[i][1],16),int(marks[i+1][1],16)
  lines=[]
  for m in re.finditer(r'/\*([0-9a-f]+)\*/\s+(.*?);',part):
   pc=int(m[1],16)
   if lo<pc<hi:lines.append((pc,m[2].strip()))
  assert not any(re.search(r'\b(?:LDG|STG|UTMALD|UTMAST|LDL|STL)\b',x) for _,x in lines),(name,'timed global/local')
  backs=[]
  for pc,ins in lines:
   m=re.search(r'\bBRA(?:\.U)?\b.*?(0x[0-9a-f]+)$',ins)
   if m and lo<int(m[1],16)<=pc:backs.append((int(m[1],16),pc))
  assert backs,(name,lo,hi)
  a,b=max(backs,key=lambda x:x[1]-x[0]);body='\n'.join(x for pc,x in lines if a<=pc<=b)
  intervals.append(dict(start=hex(lo),end=hex(hi),loop_start=hex(a),loop_end=hex(b),body=body))
 mma=[v for v in intervals if 'UTCHMMA' in v['body']];assert len(mma)==1,name
 assert len(re.findall(r'\bUTCHMMA(?:\.WS)?\s',mma[0]['body']))==32,name
 assert 'ELECT' not in mma[0]['body'],(name,'per-MMA election')
 mem=[v for v in intervals if v not in mma][0]
 op=['LDS.128','STS.128','LDTM.x32','STTM.x32','IADD3'][bg]
 count=len(re.findall(r'\b'+re.escape(op)+r'\s',mem['body']));assert count >= (8 if bg<2 else 2 if bg<4 else 1),(name,op,count)
 if bg<4:
  assert count % (8 if bg<2 else 2)==0,(name,op,count)
  mem['unrolled_transfer_factor']=count//(8 if bg<2 else 2)
 else:
  assert not re.search(r'\b(?:LDS|STS|LDTM|STTM)\b',mem['body']),name
 report[name]=dict(bg=bg,intervals=intervals)
assert len(report)==15,len(report)
resources=(p/'artifacts'/(stem+'.resources')).read_text();assert all(int(n)==0 for n in re.findall(r'(?:STACK|LOCAL):(\d+)',resources))
(p/'artifacts'/(stem+'.audit.json')).write_text(json.dumps(report,indent=2));print('PASS',len(report),'kernels: clock/backedge/opcode/no-spill audit')
