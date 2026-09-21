from pathlib import Path
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parents[3]
DEST = ROOT / "tmp" / "person-eval-videos"
BASE = "https://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1"
CLIPS = {
    "Walk1.mpg": f"{BASE}/Walk1/Walk1.mpg",
    "Browse1.mpg": f"{BASE}/Browse1/Browse1.mpg",
    "Meet_WalkTogether1.mpg": f"{BASE}/Meet_WalkTogether1/Meet_WalkTogether1.mpg",
    "Rest_FallOnFloor.mpg": f"{BASE}/Rest_FallOnFloor/Rest_FallOnFloor.mpg",
    "Fight_OneManDown.mpg": f"{BASE}/Fight_OneManDown/Fight_OneManDown.mpg",
    "Fight_Chase.mpg": f"{BASE}/Fight_Chase/Fight_Chase.mpg",
    "Fight_RunAway1.mpg": f"{BASE}/Fight_RunAway1/Fight_RunAway1.mpg",
    "LeftBag.mpg": f"{BASE}/LeftBag/LeftBag.mpg",
}

DEST.mkdir(parents=True, exist_ok=True)
for name, url in CLIPS.items():
    path = DEST / name
    if path.exists() and path.stat().st_size > 100_000:
        print(f"skip {name} ({path.stat().st_size})")
        continue
    print(f"GET {name}")
    urlretrieve(url, path)
    print(f"OK {name} {path.stat().st_size}")
