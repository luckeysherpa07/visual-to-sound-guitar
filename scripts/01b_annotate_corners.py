"""Click 4 fretboard corners per frame; the 126-point grid is generated from them.

Click order (physical, not image position):
    1. nut, high-E string     2. nut, low-E string
    3. fret F, high-E         4. fret F, low-E
F = 12 by default (double-dot marker, always visible; on classical guitars the
neck/body joint). Fret 20 is then extrapolated with the fret formula. Press f to
switch to clicking fret 20 directly (e.g. when fret 12 is under the hand).
Records always store the nut + fret-20 `corners`; the raw `clicks` and
`clicked_fret` are kept for traceability.

Keys:  enter/space = accept   u = undo click   r = reset   f = toggle fret 12 / 20
       s = skip (fretboard not usable, saved as visible=false)   q = quit
Progress is appended to the output file, so you can stop and resume.
Over SSH (no desktop window reaches you) add --web and open http://localhost:8765/;
VS Code forwards the port. Same keys, minus q (just close the tab / Ctrl+C the script).

Examples:
    # one random frame per GuiFiR-Vid train clip
    python scripts/01b_annotate_corners.py --guifir-split train --per-clip 1 \
        --out data/labels/fretboard_corners/guifir_train.jsonl
    # any folder of frames (e.g. GAPS train-split performers only!)
    python scripts/01b_annotate_corners.py --image-glob "data/raw/gaps/frames/**/*.png" \
        --source gaps --out data/labels/fretboard_corners/gaps.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from guitar2midi.data import guifir
from guitar2midi.preprocessing.fretboard_grid import (
    corners_from_clicks, corners_valid, draw_grid, grid_from_corners)

DATA_ROOT = Path("data")
MAX_W, MAX_H = 1600, 900
FRET_CHOICES = (12, 20)
MODE = {"fret": 12}  # far fret clicked; persists across frames, toggled with f


def prompts(fret: int) -> list[str]:
    return ["1: nut / high-E", "2: nut / low-E", f"3: fret {fret} / high-E", f"4: fret {fret} / low-E"]


def resolve(clicks, fret: int) -> np.ndarray | None:
    """Nut + fret-20 corners from 4 clicks on nut + ``fret``, or None if invalid."""
    clicks = np.asarray(clicks, dtype=np.float64)
    if clicks.shape != (4, 2) or not corners_valid(clicks):
        return None
    corners = corners_from_clicks(clicks, fret)
    return corners if corners_valid(corners) else None


def build_queue(args) -> list[dict]:
    rng = random.Random(args.seed)
    items = []
    if args.guifir_split:
        by_clip = defaultdict(list)
        for rec in guifir.load_split(args.guifir_split):
            by_clip[guifir.clip_id(rec)].append(rec)
        for cid in sorted(by_clip):
            for rec in rng.sample(by_clip[cid], min(args.per_clip, len(by_clip[cid]))):
                items.append(dict(
                    path=guifir.frame_path(rec, args.guifir_split),
                    source=f"guifir_{args.guifir_split}", clip_id=cid))
    else:
        for p in sorted(Path().glob(args.image_glob)):
            items.append(dict(path=p, source=args.source, clip_id=p.parent.name))
    rng.shuffle(items)
    return items[: args.limit] if args.limit else items


_on_top: set[str] = set()


def keep_on_top(title: str) -> None:
    """Ask the X11 window manager to keep the window above others.

    cv2.WND_PROP_TOPMOST is ignored by OpenCV's Qt backend on Linux, so this
    sends the EWMH _NET_WM_STATE_ABOVE request via python-xlib (pip install -e .[annotate]).
    """
    if title in _on_top:
        return
    _on_top.add(title)
    try:
        from Xlib import X, display
        from Xlib.protocol import event
    except ImportError:
        print("python-xlib not installed; window will not stay on top")
        return
    try:
        d = display.Display()
        root = d.screen().root

        def find(w):
            try:
                if w.get_wm_name() == title:
                    return w
            except Exception:
                pass
            for c in w.query_tree().children:
                if (r := find(c)) is not None:
                    return r
            return None

        win = find(root)
        if win is None:
            return
        ev = event.ClientMessage(window=win, client_type=d.intern_atom("_NET_WM_STATE"),
                                 data=(32, [1, d.intern_atom("_NET_WM_STATE_ABOVE"), 0, 1, 0]))
        root.send_event(ev, event_mask=X.SubstructureRedirectMask | X.SubstructureNotifyMask)
        d.flush()
        d.close()
    except Exception as e:  # never block annotation because of window placement
        print("could not keep window on top:", e)


def annotate(img: np.ndarray, title: str) -> dict | None | str:
    """Returns dict(corners, clicks, fret), None (skip) or 'quit'."""
    scale = min(MAX_W / img.shape[1], MAX_H / img.shape[0], 1.0)
    pts: list[tuple[float, float]] = []

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x / scale, y / scale))

    cv2.namedWindow(title)
    cv2.setMouseCallback(title, on_mouse)
    while True:
        view = img.copy()
        fret = MODE["fret"]
        corners = resolve(pts, fret) if len(pts) == 4 else None
        valid = corners is not None
        if valid:
            view = draw_grid(view, grid_from_corners(corners), thickness=2)
        for i, p in enumerate(pts):
            cv2.circle(view, (int(p[0]), int(p[1])), 6, (0, 0, 255), -1)
            cv2.putText(view, str(i + 1), (int(p[0]) + 8, int(p[1]) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        view = cv2.resize(view, None, fx=scale, fy=scale)
        if len(pts) < 4:
            msg = prompts(fret)[len(pts)]
        else:
            msg = "enter=accept  u=undo  r=reset" if valid else "INVALID (crossed) - u=undo r=reset"
        cv2.putText(view, msg + f"   [f=fret {fret}->{32 - fret} s=skip q=quit]", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow(title, view)
        key = cv2.waitKey(20) & 0xFF
        keep_on_top(title)  # after the first imshow, once the window is mapped
        if key == ord("q"):
            return "quit"
        if key == ord("s"):
            return None
        if key == ord("u") and pts:
            pts.pop()
        if key == ord("r"):
            pts.clear()
        if key == ord("f"):
            MODE["fret"] = 32 - fret  # 12 <-> 20
        if key in (13, 32) and valid:
            return dict(corners=corners.tolist(), clicks=[list(p) for p in pts], fret=fret)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--guifir-split", choices=["train"], help="sample frames from GuiFiR-Vid (train only: val/test are evaluation)")
    src.add_argument("--image-glob", help="glob of frame images")
    ap.add_argument("--source", default="images", help="source name for --image-glob")
    ap.add_argument("--per-clip", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--web", type=int, nargs="?", const=8765, default=0, metavar="PORT",
                    help="annotate in the browser instead of an OpenCV window (for SSH; default port 8765)")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.out.exists():
        done = {json.loads(l)["image"] for l in args.out.read_text().splitlines() if l.strip()}

    queue = [it for it in build_queue(args) if str(it["path"].relative_to(DATA_ROOT)) not in done]
    print(f"{len(done)} done, {len(queue)} to annotate")
    if args.web:
        serve_web(queue, args.out, len(done), args.web)
        return
    with open(args.out, "a") as f:
        for n, it in enumerate(queue):
            img = cv2.imread(str(it["path"]))
            if img is None:
                print("cannot read", it["path"])
                continue
            res = annotate(img, "corners")
            if res == "quit":
                break
            f.write(json.dumps(make_record(it, img.shape, res)) + "\n")
            f.flush()
            print(f"[{n + 1}/{len(queue)}] {it['path'].name}: {'ok' if res else 'skipped'}")
    cv2.destroyAllWindows()


def make_record(it: dict, shape: tuple, ann: dict | None) -> dict:
    """``ann`` = dict(corners, clicks, fret) from annotate()/the web page, or None (skipped)."""
    return dict(
        image=str(it["path"].relative_to(DATA_ROOT)),
        source=it["source"], clip_id=it["clip_id"],
        frame_width=shape[1], frame_height=shape[0],
        corners=ann["corners"] if ann else None, visible=ann is not None,
        clicks=ann["clicks"] if ann else None, clicked_fret=ann["fret"] if ann else None,
        label_source="manual",
    )


# ---------------------------------------------------------------- browser mode
# For working over SSH: the page is served on localhost and VS Code forwards the port.

WEB_PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Fretboard corners</title>
<style>
 body{margin:0;background:#1e1e1e;color:#ddd;font:15px system-ui,sans-serif}
 #bar{padding:8px 12px;display:flex;gap:16px;align-items:center;flex-wrap:wrap}
 #prompt{font-weight:600;color:#ffd75e} #bad{color:#ff6b6b;font-weight:600}
 canvas{display:block;margin:0 auto;cursor:crosshair}
 kbd{background:#333;border-radius:3px;padding:1px 5px}
</style></head><body>
<div id="bar"><span id="prog"></span><span id="mode"></span><span id="prompt"></span><span id="bad"></span>
<span style="margin-left:auto"><kbd>Enter</kbd>/<kbd>Space</kbd> accept · <kbd>u</kbd> undo ·
<kbd>r</kbd> reset · <kbd>f</kbd> fret 12/20 · <kbd>s</kbd> skip (unusable)</span></div>
<canvas id="c"></canvas>
<script>
const PROMPTS=f=>["1: nut / high-E","2: nut / low-E",`3: fret ${f} / high-E`,`4: fret ${f} / low-E`];
const cv=document.getElementById('c'),ctx=cv.getContext('2d');
let item=null,img=new Image(),pts=[],grid=null,valid=false,scale=1,busy=false,fret=12;
async function next(){
  const r=await (await fetch('api/next')).json();
  if(r.finished){document.getElementById('prog').textContent='All done ('+r.done+' annotated). You can close this tab.';
    document.getElementById('prompt').textContent='';cv.width=0;return;}
  item=r;pts=[];grid=null;valid=false;
  img=new Image();img.onload=()=>{fit();draw();busy=false};img.src='img/'+r.index+'?t='+Date.now();
  document.getElementById('prog').textContent=`[${r.position}/${r.total}]  ${r.name}  (${r.done} saved)`;
}
function fit(){scale=Math.min((innerWidth-24)/img.width,(innerHeight-60)/img.height);
  cv.width=Math.round(img.width*scale);cv.height=Math.round(img.height*scale);}
function line(a,b,col,w){ctx.strokeStyle=col;ctx.lineWidth=w;ctx.beginPath();
  ctx.moveTo(a[0]*scale,a[1]*scale);ctx.lineTo(b[0]*scale,b[1]*scale);ctx.stroke();}
function draw(){
  if(!img.width)return;ctx.drawImage(img,0,0,cv.width,cv.height);
  if(grid&&valid){const g=(s,f)=>grid[s*21+f];
    for(let f=0;f<21;f++)line(g(0,f),g(5,f),f==0?'#ff3b3b':(f==fret?'#ffd75e':'#3b8bff'),f==0||f==fret?3:1.5);
    for(let s=0;s<6;s++)line(g(s,0),g(s,20),'#3bff6b',1.5);}
  pts.forEach((p,i)=>{ctx.fillStyle='#ff3030';ctx.beginPath();ctx.arc(p[0]*scale,p[1]*scale,5,0,7);ctx.fill();
    ctx.font='bold 18px sans-serif';ctx.fillText(i+1,p[0]*scale+8,p[1]*scale-8);});
  document.getElementById('mode').textContent=`clicking fret ${fret}`;
  document.getElementById('prompt').textContent=pts.length<4?PROMPTS(fret)[pts.length]:(valid?'Enter = accept':'');
  document.getElementById('bad').textContent=pts.length==4&&!valid?'INVALID (crossed) – u = undo, r = reset':'';
}
async function update(){grid=null;valid=false;
  if(pts.length==4){const r=await (await fetch('api/grid',{method:'POST',body:JSON.stringify({clicks:pts,fret})})).json();
    valid=r.valid;grid=r.grid;}
  draw();}
cv.addEventListener('mousedown',e=>{if(busy||pts.length>=4)return;const b=cv.getBoundingClientRect();
  pts.push([(e.clientX-b.left)/scale,(e.clientY-b.top)/scale]);update();});
async function save(clicks){busy=true;
  await fetch('api/save',{method:'POST',body:JSON.stringify({index:item.index,clicks,fret})});next();}
addEventListener('keydown',e=>{if(!item||busy)return;
  if(e.key=='u'&&pts.length){pts.pop();update();}
  else if(e.key=='r'){pts=[];update();}
  else if(e.key=='f'){fret=32-fret;update();}
  else if(e.key=='s')save(null);
  else if((e.key=='Enter'||e.key==' ')&&valid){e.preventDefault();save(pts);}});
addEventListener('resize',()=>{fit();draw()});
next();
</script></body></html>"""


def serve_web(queue: list[dict], out: Path, n_done: int, port: int) -> None:
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    state = dict(cur=0, done=n_done)
    lock = threading.Lock()

    def current() -> int | None:
        return state["cur"] if state["cur"] < len(queue) else None

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # keep the terminal quiet
            pass

        def send(self, body: bytes, ctype: str, code: int = 200):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, obj):
            self.send(json.dumps(obj).encode(), "application/json")

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                return self.send(WEB_PAGE.encode(), "text/html; charset=utf-8")
            if self.path == "/api/next":
                with lock:
                    i = current()
                    if i is None:
                        return self.send_json(dict(finished=True, done=state["done"]))
                    return self.send_json(dict(index=i, position=i + 1, total=len(queue),
                                               done=state["done"], name=queue[i]["path"].name))
            if self.path.startswith("/img/"):
                i = int(self.path[5:].split("?")[0])
                return self.send(queue[i]["path"].read_bytes(), "image/png")
            self.send(b"not found", "text/plain", 404)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
            fret = body.get("fret", 20)
            if fret not in FRET_CHOICES:
                return self.send(b"bad fret", "text/plain", 400)
            if self.path == "/api/grid":
                corners = resolve(body["clicks"], fret)
                return self.send_json(dict(valid=corners is not None,
                                           grid=None if corners is None else grid_from_corners(corners).tolist()))
            if self.path == "/api/save":
                with lock:
                    i = current()
                    if i is None or body["index"] != i:
                        return self.send_json(dict(ok=False))  # stale page / double submit
                    ann = None
                    if body["clicks"] is not None:
                        corners = resolve(body["clicks"], fret)
                        if corners is None:
                            return self.send_json(dict(ok=False))
                        ann = dict(corners=corners.tolist(), clicks=body["clicks"], fret=fret)
                    it = queue[i]
                    img = cv2.imread(str(it["path"]))
                    if img is not None:
                        with open(out, "a") as f:
                            f.write(json.dumps(make_record(it, img.shape, ann)) + "\n")
                        state["done"] += 1
                        print(f"[{i + 1}/{len(queue)}] {it['path'].name}: "
                              f"{f'ok (fret {fret})' if ann else 'skipped'}", flush=True)
                    else:
                        print("cannot read", it["path"], flush=True)
                    state["cur"] += 1
                return self.send_json(dict(ok=True))
            self.send(b"not found", "text/plain", 404)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Open http://localhost:{port}/ (VS Code forwards the port over SSH). Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
