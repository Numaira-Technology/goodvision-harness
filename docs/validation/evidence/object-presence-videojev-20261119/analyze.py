import gzip, json, collections, sys
from pathlib import Path
from tools.unified_benchmark.latency import nearest_rank
S = sys.argv[1]
def load(name):
    plain, packed = Path(S) / f"{name}.jsonl", Path(S) / f"{name}.jsonl.gz"
    handle = open(plain) if plain.exists() else gzip.open(packed, "rt")
    with handle:
        return [json.loads(l) for l in handle]
rows=load("live")
off=load("offline")
def pct(v): v=sorted(v); return f"P50 {nearest_rank(v,.5):.2f}s P95 {nearest_rank(v,.95):.2f}s max {max(v):.2f}s n={len(v)}"
for tag in ("live_obj","live_par"):
    obj=[r for r in rows if r["kind"]=="object" and r["tag"]==tag]
    gaps=[]; source=[]; byclip=collections.defaultdict(list)
    for r in obj: byclip[r["clip"]].append(r)
    for rs in byclip.values():
        rs.sort(key=lambda r:r["wall"])
        # wall = when the result completed; timestamp_seconds = judged source frame.
        gaps+= [b["wall"]-a["wall"] for a,b in zip(rs,rs[1:])]
        source+= [round(b["timestamp_seconds"]-a["timestamp_seconds"],3) for a,b in zip(rs,rs[1:])]
    print("==",tag)
    print(" object completion interval", pct(gaps))
    print(" object source-frame interval", pct(source), "counts", sorted(collections.Counter(source).items()))
    print(" object transport", pct([r["transport_seconds"] for r in obj]))
    print(" object errors", dict(collections.Counter(r["error"] for r in obj if r["error"])), "of", len(obj))
    ev=[r for r in rows if r["kind"]=="event" and r["tag"]==tag]
    if ev:
        egaps=[]; eb=collections.defaultdict(list)
        for r in ev: eb[r["clip"]].append(r["wall"])
        for w in eb.values(): w.sort(); egaps+=[b-a for a,b in zip(w,w[1:])]
        print(" event interval", pct(egaps)); print(" event latency", pct([r["seconds"] for r in ev if r["status"]==200]))
        print(" event errors", dict(collections.Counter(r["error"] for r in ev if r["error"])), "of", len(ev))
    for oid,tk in (("knife","knife"),("alcohol_bottle","alcohol")):
        tp=fp=fn=tn=0; detail=[]
        for clip,rs in byclip.items():
            truth=rs[0]["truth"][tk]
            hit=any(o["present"] for r in rs for o in r["objects"] if o["id"]==oid)
            mx=max(o["present_probability"] or 0 for r in rs for o in r["objects"] if o["id"]==oid)
            tp+=truth and hit; fn+=truth and not hit; fp+=(not truth) and hit; tn+=(not truth) and not hit
            if truth or hit or mx>=0.5: detail.append(f"{clip}:{'Y' if truth else '-'}{'hit' if hit else 'miss'}@{mx:.2f}")
        print(f" {oid:15s} @0.95  TP {tp} FN {fn} FP {fp} TN {tn}  ", " ".join(detail))
print("== descriptive gate sweep (clip max prob; tiny set, NOT calibration)")
for tag in ("off1","off25","live_obj","live_par"):
    src=off if tag.startswith("off") else [r for r in rows if r["kind"]=="object"]
    m=collections.defaultdict(float)
    for r in src:
        if r["tag"]!=tag: continue
        for o in r["objects"]:
            k=(r["clip"],o["id"],r["truth"]["knife" if o["id"]=="knife" else "alcohol"])
            m[k]=max(m[k],o["present_probability"] or 0)
    for g in (0.7,0.8,0.9,0.95):
        out=[]
        for oid in ("knife","alcohol_bottle"):
            tp=sum(1 for (c,o,t),p in m.items() if o==oid and t and p>=g); fn=sum(1 for (c,o,t),p in m.items() if o==oid and t and p<g)
            fp=sum(1 for (c,o,t),p in m.items() if o==oid and not t and p>=g)
            out.append(f"{oid} TP{tp} FN{fn} FP{fp}")
        print(f" {tag:9s} gate {g}: "+" | ".join(out))
