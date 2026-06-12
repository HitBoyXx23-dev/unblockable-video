from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import struct
import subprocess
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import cv2
from fastapi import Body, FastAPI, File, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

ROOT = Path(__file__).resolve().parent
INDEX_FILE = ROOT / "index.html"
API_DOC_FILE = ROOT / "api_doc.html"
MEDIA_DIR = ROOT / "media"
MEDIA_DIR.mkdir(exist_ok=True)
DEFAULT_VIDEO = MEDIA_DIR / "input.mp4"
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}

STATE: dict[str, Any] = {
    "source": str(DEFAULT_VIDEO) if DEFAULT_VIDEO.exists() else "",
    "width": int(os.getenv("UNBLOCKABLE_WIDTH", os.getenv("ASCILINE_WIDTH", "960"))),
    "fps": int(os.getenv("UNBLOCKABLE_FPS", os.getenv("ASCILINE_FPS", "60"))),
    "quality": int(os.getenv("UNBLOCKABLE_QUALITY", os.getenv("ASCILINE_JPEG_QUALITY", "78"))),
}

app = FastAPI(title="Unblockable Video")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/media", StaticFiles(directory=str(MEDIA_DIR)), name="media")


def safe_name(name: str) -> str:
    stem = Path(name).stem.strip() or "video"
    suffix = Path(name).suffix.lower()
    stem = re.sub(r"[^a-zA-Z0-9._-]+", "-", stem).strip(".-_") or "video"
    if suffix not in ALLOWED_EXTENSIONS:
        suffix = ".mp4"
    candidate = f"{stem}{suffix}"
    index = 1
    while (MEDIA_DIR / candidate).exists():
        candidate = f"{stem}-{index}{suffix}"
        index += 1
    return candidate


def clean_source(value: str | None) -> str:
    if not value:
        return str(STATE.get("source", ""))
    source = value.strip().strip('"').strip("'")
    if not source:
        return ""
    parsed = urlparse(source)
    if parsed.scheme in {"http", "https", "rtmp", "rtsp"}:
        return source
    if source.startswith("/media/"):
        return str((MEDIA_DIR / source.split("/media/", 1)[1]).resolve())
    path = Path(source).expanduser()
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    return str(path)


def public_source(source: str) -> str:
    try:
        path = Path(source).resolve()
        if MEDIA_DIR.resolve() in path.parents or path == MEDIA_DIR.resolve():
            return "/media/" + path.name
    except Exception:
        pass
    return source


def source_exists(source: str) -> bool:
    parsed = urlparse(source)
    if parsed.scheme in {"http", "https", "rtmp", "rtsp"}:
        return True
    return Path(source).exists()


def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def has_ffprobe() -> bool:
    return shutil.which("ffprobe") is not None


def get_video_meta(source: str) -> dict[str, Any]:
    data = {
        "name": "Unblockable Video",
        "source": public_source(source) if source else "",
        "resolvedSource": source,
        "hasSource": bool(source),
        "sourceExists": source_exists(source) if source else False,
        "width": 0,
        "height": 0,
        "fps": STATE["fps"],
        "duration": 0,
        "audio": has_ffmpeg(),
        "serverWidth": STATE["width"],
        "serverFps": STATE["fps"],
        "serverQuality": STATE["quality"],
        "mode": "canvas-websocket",
    }
    if not source:
        return data
    if has_ffprobe():
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=width,height,r_frame_rate:format=duration",
                    "-of",
                    "json",
                    source,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=10,
            )
            payload = json.loads(result.stdout or "{}")
            streams = payload.get("streams") or []
            if streams:
                stream = streams[0]
                data["width"] = int(stream.get("width") or 0)
                data["height"] = int(stream.get("height") or 0)
                rate = str(stream.get("r_frame_rate") or "0/1")
                top, bottom = rate.split("/") if "/" in rate else (rate, "1")
                if float(bottom or 1):
                    data["fps"] = round(float(top) / float(bottom), 3)
            duration = float((payload.get("format") or {}).get("duration") or 0)
            data["duration"] = round(duration, 3)
            return data
        except Exception:
            pass
    cap = cv2.VideoCapture(source)
    if cap.isOpened():
        fps = cap.get(cv2.CAP_PROP_FPS) or STATE["fps"]
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        data["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        data["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        data["fps"] = round(float(fps), 3)
        data["duration"] = round(float(frames / fps), 3) if fps and frames > 0 else 0
        cap.release()
    return data


def audio_filter(speed: float) -> list[str]:
    value = max(0.25, min(4.0, float(speed)))
    filters: list[str] = []
    while value > 2.0:
        filters.append("atempo=2.0")
        value /= 2.0
    while value < 0.5:
        filters.append("atempo=0.5")
        value /= 0.5
    filters.append(f"atempo={value:.4f}")
    return ["-filter:a", ",".join(filters)]


def read_jpeg_dimensions(data: bytes) -> tuple[int, int]:
    index = 2
    length = len(data)
    while index + 9 < length:
        if data[index] != 255:
            index += 1
            continue
        marker = data[index + 1]
        if marker in {192, 193, 194}:
            height = int.from_bytes(data[index + 5:index + 7], "big")
            width = int.from_bytes(data[index + 7:index + 9], "big")
            return width, height
        segment = int.from_bytes(data[index + 2:index + 4], "big")
        index += 2 + segment
    return 0, 0


def pack_frame(width: int, height: int, frame_no: int, pts_ms: int, jpg: bytes) -> bytes:
    return b"IMG1" + struct.pack("<HHIII", width, height, frame_no, pts_ms, len(jpg)) + jpg


def extract_jpegs(buffer: bytearray) -> list[bytes]:
    frames: list[bytes] = []
    while True:
        start = buffer.find(b"\xff\xd8")
        if start < 0:
            if len(buffer) > 65536:
                del buffer[:-2]
            return frames
        end = buffer.find(b"\xff\xd9", start + 2)
        if end < 0:
            if start > 0:
                del buffer[:start]
            return frames
        end += 2
        frames.append(bytes(buffer[start:end]))
        del buffer[:end]


def ffmpeg_video_command(source: str, start: float, speed: float, width: int, fps: int, quality: int) -> list[str]:
    playback_speed = max(0.25, min(4.0, speed))
    video_filter = f"setpts=PTS/{playback_speed:.4f},fps={fps},scale={width}:-2:flags=fast_bilinear"
    qscale = max(2, min(24, round((100 - quality) / 4) + 2))
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        str(max(0.0, start)),
        "-i",
        source,
        "-an",
        "-vf",
        video_filter,
        "-q:v",
        str(qscale),
        "-f",
        "image2pipe",
        "-vcodec",
        "mjpeg",
        "-",
    ]


def ffmpeg_audio_command(source: str, start: float, speed: float) -> list[str]:
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        str(max(0.0, start)),
        "-i",
        source,
        "-vn",
        *audio_filter(speed),
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-ac",
        "2",
        "-ar",
        "48000",
        "-",
    ]


@app.get("/")
def index() -> FileResponse:
    return FileResponse(INDEX_FILE)


@app.get("/api_doc.html")
def api_doc() -> FileResponse:
    return FileResponse(API_DOC_FILE)


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse({"ok": True, "name": "Unblockable Video"})


@app.get("/api/status")
def status(src: str = "") -> JSONResponse:
    source = clean_source(src) if src else clean_source(None)
    return JSONResponse(get_video_meta(source))


@app.get("/api/uploads")
def uploads() -> JSONResponse:
    files = []
    for path in sorted(MEDIA_DIR.iterdir()):
        if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS:
            files.append({"name": path.name, "src": f"/media/{path.name}", "bytes": path.stat().st_size})
    return JSONResponse({"ok": True, "files": files})


@app.post("/api/source")
def set_source(payload: dict[str, Any] = Body(default_factory=dict)) -> JSONResponse:
    source = clean_source(str(payload.get("src", "")))
    if not source:
        return JSONResponse({"ok": False, "error": "Missing src"}, status_code=400)
    if not source_exists(source):
        return JSONResponse({"ok": False, "error": "Source not found"}, status_code=404)
    STATE["source"] = source
    return JSONResponse({"ok": True, "meta": get_video_meta(source)})


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...), autoplay: bool = Query(True)) -> JSONResponse:
    filename = safe_name(file.filename or "video.mp4")
    destination = MEDIA_DIR / filename
    size = 0
    try:
        with destination.open("wb") as handle:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                handle.write(chunk)
    except Exception as error:
        if destination.exists():
            destination.unlink()
        return JSONResponse({"ok": False, "error": str(error)}, status_code=500)
    if size <= 0:
        destination.unlink(missing_ok=True)
        return JSONResponse({"ok": False, "error": "Empty upload"}, status_code=400)
    source = str(destination.resolve())
    if autoplay:
        STATE["source"] = source
    return JSONResponse({"ok": True, "src": f"/media/{filename}", "bytes": size, "meta": get_video_meta(source)})


@app.delete("/api/upload/{name}")
def delete_upload(name: str) -> JSONResponse:
    path = MEDIA_DIR / Path(name).name
    if not path.exists() or path.suffix.lower() not in ALLOWED_EXTENSIONS:
        return JSONResponse({"ok": False, "error": "File not found"}, status_code=404)
    path.unlink()
    if clean_source(None) == str(path.resolve()):
        STATE["source"] = ""
    return JSONResponse({"ok": True})


@app.get("/api/embed")
def embed_code(src: str = Query(...), width: str = "960px", height: str = "540px") -> HTMLResponse:
    safe_src = src.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    safe_width = width.replace('"', "")
    safe_height = height.replace('"', "")
    code = f'<src width="{safe_width}" height="{safe_height}">{safe_src}</src><script src="/api/player.js"></script>'
    return HTMLResponse(code)


@app.get("/api/player.js")
def player_js() -> HTMLResponse:
    script = r'''
(function(){
function baseFromScript(){
var scripts=document.getElementsByTagName('script');
for(var i=scripts.length-1;i>=0;i--){
var value=scripts[i].src||'';
var marker='/api/player.js';
var index=value.indexOf(marker);
if(index>-1){return value.slice(0,index)}
}
return location.origin
}
function sourceFromNode(node){
return (node.getAttribute('url')||node.getAttribute('src')||node.getAttribute('data-src')||node.textContent||'').trim()
}
function replaceNode(node,base){
var source=sourceFromNode(node);
if(!source){return}
var frame=document.createElement('iframe');
frame.src=base+'/?embed=1&src='+encodeURIComponent(source);
frame.style.width=node.getAttribute('width')||'960px';
frame.style.height=node.getAttribute('height')||'540px';
frame.style.border='0';
frame.style.background='#000';
frame.allow='fullscreen; autoplay';
node.replaceWith(frame)
}
var base=baseFromScript();
var srcNodes=Array.prototype.slice.call(document.getElementsByTagName('src'));
var uvNodes=Array.prototype.slice.call(document.getElementsByTagName('unblockable-video'));
srcNodes.concat(uvNodes).forEach(function(node){replaceNode(node,base)})
})();
'''.strip()
    return HTMLResponse(script, media_type="application/javascript")


@app.websocket("/api/frames")
async def frames(ws: WebSocket, src: str = "", start: float = 0.0, speed: float = 1.0, width: int = 0, fps: int = 0, quality: int = 0) -> None:
    await ws.accept()
    source = clean_source(src)
    if not source or not source_exists(source):
        await ws.close()
        return
    target_width = max(160, min(1920, width or STATE["width"]))
    target_fps = max(1, min(120, fps or STATE["fps"]))
    frame_quality = max(35, min(95, quality or STATE["quality"]))
    playback_speed = max(0.25, min(4.0, float(speed)))
    if has_ffmpeg():
        command = ffmpeg_video_command(source, start, playback_speed, target_width, target_fps, frame_quality)
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        buffer = bytearray()
        frame_no = 0
        clock = time.perf_counter()
        try:
            while True:
                chunk = proc.stdout.read(65536) if proc.stdout else b""
                if not chunk:
                    break
                buffer.extend(chunk)
                jpgs = extract_jpegs(buffer)
                if len(jpgs) > 3:
                    jpgs = jpgs[-3:]
                for jpg in jpgs:
                    width_value, height_value = read_jpeg_dimensions(jpg)
                    pts_ms = int((start + frame_no * playback_speed / target_fps) * 1000)
                    await ws.send_bytes(pack_frame(width_value, height_value, frame_no, pts_ms, jpg))
                    frame_no += 1
                    delay = clock + frame_no / target_fps - time.perf_counter()
                    if delay > 0:
                        await asyncio.sleep(delay)
                    elif delay < -0.5:
                        clock = time.perf_counter() - frame_no / target_fps
        except WebSocketDisconnect:
            pass
        finally:
            proc.kill()
        return
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        await ws.close()
        return
    native_fps = cap.get(cv2.CAP_PROP_FPS) or target_fps
    frame_no = 0
    clock = time.perf_counter()
    try:
        while True:
            video_time = start + frame_no * playback_speed / target_fps
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(video_time * native_fps))
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            target_height = max(1, round(h * target_width / w))
            resized = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
            ok, encoded = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), frame_quality])
            if ok:
                await ws.send_bytes(pack_frame(target_width, target_height, frame_no, int(video_time * 1000), encoded.tobytes()))
            frame_no += 1
            delay = clock + frame_no / target_fps - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
    except WebSocketDisconnect:
        pass
    finally:
        cap.release()


@app.websocket("/api/audio")
async def audio(ws: WebSocket, src: str = "", start: float = 0.0, speed: float = 1.0) -> None:
    await ws.accept()
    source = clean_source(src)
    if not source or not source_exists(source) or not has_ffmpeg():
        await ws.send_json({"type": "audio-missing"})
        await ws.close()
        return
    command = ffmpeg_audio_command(source, start, max(0.25, min(4.0, float(speed))))
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    await ws.send_json({"type": "audio-meta", "sampleRate": 48000, "channels": 2, "start": start, "speed": speed})
    rate = 48000
    channels = 2
    bytes_per_sample = 2
    chunk_size = 16384
    sent = 0
    clock = time.perf_counter()
    try:
        while True:
            data = proc.stdout.read(chunk_size) if proc.stdout else b""
            if not data:
                break
            await ws.send_bytes(data)
            sent += len(data)
            duration = sent / (rate * channels * bytes_per_sample)
            delay = clock + duration - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
    except WebSocketDisconnect:
        pass
    finally:
        proc.kill()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", nargs="?", default="")
    parser.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    parser.add_argument("--width", type=int, default=STATE["width"])
    parser.add_argument("--fps", type=int, default=STATE["fps"])
    parser.add_argument("--quality", type=int, default=STATE["quality"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.video:
        STATE["source"] = clean_source(args.video)
    STATE["width"] = args.width
    STATE["fps"] = args.fps
    STATE["quality"] = args.quality
    print(f"Unblockable Video running at http://127.0.0.1:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, reload=False)
