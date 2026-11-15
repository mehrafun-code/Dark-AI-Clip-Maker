import json
import re
import subprocess
from pathlib import Path


# ---------- 1. Download ----------
def _clean(msg):
    """Error message se rang ke ajeeb characters hatao."""
    return re.sub(r"\x1b\[[0-9;]*m|\[[0-9;]*m", "", str(msg)).replace("\ufffd", "").strip()


def download_video(url, out_dir, cookies_browser=None):
    import yt_dlp
    out_dir = Path(out_dir)
    # Pehle behtar quality + merge (mkv sab codecs leta hai), phir simple single-file format
    attempts = [
        {"format": "bv*[height<=1080][vcodec^=avc1]+ba[ext=m4a]/bv*[height<=1080]+ba/b[height<=1080]/b",
         "merge_output_format": "mkv"},
        {"format": "b[height<=720]/b"},
        {"format": "worst"},
    ]
    last_err = None
    for extra in attempts:
        for old in out_dir.glob("source.*"):
            try:
                old.unlink()
            except OSError:
                pass
        opts = {
            "outtmpl": str(out_dir / "source.%(ext)s"),
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "retries": 5,
            "fragment_retries": 5,
            "concurrent_fragment_downloads": 4,
            **extra,
        }
        if cookies_browser:
            opts["cookiesfrombrowser"] = (cookies_browser,)
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=True)
            files = [p for p in out_dir.glob("source.*") if p.suffix not in (".part", ".ytdl")]
            if files:
                files.sort(key=lambda p: p.stat().st_size, reverse=True)
                return files[0], info.get("title", "video")
        except Exception as e:  # agla format try karo
            last_err = e
            low = str(e).lower()
            if "confirm your age" in low or "sign in" in low:
                raise RuntimeError(
                    "Ye video age-restricted hai (YouTube login mangta hai). Ya to koi aam video use karo, "
                    "ya sidebar mein 'Browser login' chuno (us browser mein YouTube par login hona chahiye)."
                )
            if "cookie" in low or "dpapi" in low or "could not copy" in low:
                raise RuntimeError(
                    "Browser ki cookies nahi padh saka. Browser band karke dobara try karo, "
                    "ya Firefox / Edge chuno. Detail: " + _clean(e)
                )
    raise RuntimeError("Video download nahi ho saki. Link check karo (live stream ho to khatam ho chuki ho). Detail: " + _clean(last_err))


# ---------- 2. Transcribe ----------
def transcribe(video_path, model_size="small", device="cpu"):
    from faster_whisper import WhisperModel
    compute = "int8" if device == "cpu" else "float16"
    model = WhisperModel(model_size, device=device, compute_type=compute)
    segments, _ = model.transcribe(str(video_path), word_timestamps=True, vad_filter=True)
    result = []
    for s in segments:
        words = [{"w": w.word.strip(), "start": w.start, "end": w.end}
                 for w in (s.words or []) if w.word.strip()]
        result.append({"start": s.start, "end": s.end, "text": s.text.strip(), "words": words})
    return result


# ---------- 3. Best clips chunna ----------
def _snap(segments, start, end, min_d, max_d):
    """Clip ko sentence ki boundary par laata hai aur duration limit mein rakhta hai."""
    starts = [s["start"] for s in segments]
    ends = [s["end"] for s in segments]
    start = min(starts, key=lambda x: abs(x - start))
    end = min(ends, key=lambda x: abs(x - end))
    if end - start > max_d:
        valid = [e for e in ends if min_d <= e - start <= max_d]
        end = valid[-1] if valid else start + max_d
    if end - start < min_d:
        end = start + min_d
    return round(start, 2), round(end, 2)


def pick_clips_ai(segments, api_key, model, n_clips, min_d, max_d):
    import anthropic
    transcript = "\n".join(f"[{s['start']:.1f}-{s['end']:.1f}] {s['text']}" for s in segments)
    if len(transcript) > 150000:
        transcript = transcript[:150000]
    prompt = f"""Neeche ek video ka transcript timestamps ke saath hai.
Isme se {n_clips} sabse acche, viral hone wale clips chuno jo Instagram Reels / YouTube Shorts ke liye hon.
Rules:
- Har clip {min_d} se {max_d} seconds ka ho.
- Clip ka start ek mazboot hook se ho (sawal, hairan karne wali baat, bold claim).
- Clip apne aap mein mukammal ho (beech mein na shuru ho na beech mein khatam).
- Clips ek dosre se overlap na karein.
- Start/end transcript ke timestamps ke mutabiq hon.
Sirf JSON array do, aur kuch nahi. Format:
[{{"start": 12.5, "end": 58.0, "title": "chota title", "score": 92, "reason": "ek line mein wajah"}}]

Transcript:
{transcript}"""
    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=model, max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in resp.content if b.type == "text")
    m = re.search(r"\[.*\]", text, re.S)
    items = json.loads(m.group(0))
    clips = []
    for it in items:
        s, e = _snap(segments, float(it["start"]), float(it["end"]), min_d, max_d)
        clips.append({"start": s, "end": e, "title": it.get("title", "Clip"),
                      "score": it.get("score", 0), "reason": it.get("reason", "")})
    return clips


HOOK_WORDS = ["secret", "never", "mistake", "why", "how", "best", "worst", "money",
              "free", "truth", "shocking", "amazing", "biggest", "important", "kabhi",
              "raaz", "sach", "paisa", "galti", "sabse"]


def pick_clips_heuristic(segments, n_clips, min_d, max_d):
    """Bina API ke: bolne ki speed aur hook words se score."""
    candidates = []
    for i, seg in enumerate(segments):
        start = seg["start"]
        j = i
        while j < len(segments) and segments[j]["end"] - start <= max_d:
            j += 1
        j -= 1
        if j < i or segments[j]["end"] - start < min_d:
            continue
        chunk = segments[i:j + 1]
        text = " ".join(c["text"] for c in chunk).lower()
        dur = chunk[-1]["end"] - start
        words = len(text.split())
        score = words / dur
        score += 0.5 * sum(text.count(w) for w in HOOK_WORDS)
        score += 0.3 * (text.count("?") + text.count("!"))
        if "?" in chunk[0]["text"]:
            score += 1.5
        candidates.append({"start": start, "end": chunk[-1]["end"],
                           "title": chunk[0]["text"][:50], "score": round(score * 10),
                           "reason": "Heuristic score (bina AI)"})
    candidates.sort(key=lambda c: c["score"], reverse=True)
    chosen = []
    for c in candidates:
        if all(c["end"] <= x["start"] or c["start"] >= x["end"] for x in chosen):
            chosen.append(c)
        if len(chosen) == n_clips:
            break
    return chosen


# ---------- 4. Face tracking (9:16 crop ke liye) ----------
def face_center(video_path, start, end, samples=8):
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    det = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    xs = []
    for k in range(samples):
        t = start + (end - start) * (k + 0.5) / samples
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = det.detectMultiScale(gray, 1.2, 5, minSize=(int(h * 0.1), int(h * 0.1)))
        if len(faces):
            x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
            xs.append(x + fw / 2)
    cap.release()
    xs.sort()
    cx = xs[len(xs) // 2] if xs else w / 2
    return int(w), int(h), cx


# ---------- 5. Captions (ASS, word-by-word highlight) ----------
def _ass_time(t):
    t = max(t, 0)
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def make_ass(words, clip_start, clip_end, path, per_line=3, brand_text=""):
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding
Style: Default,Arial,80,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,6,2,2,60,60,520,1
Style: Brand,Arial,52,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,4,2,8,60,60,110,1

[Events]
Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
"""
    ws = [w for w in words if w["start"] >= clip_start - 0.1 and w["end"] <= clip_end + 0.1]
    lines = []
    for i in range(0, len(ws), per_line):
        group = ws[i:i + per_line]
        for k, w in enumerate(group):
            s = w["start"] - clip_start
            e = (group[k + 1]["start"] if k + 1 < len(group) else w["end"]) - clip_start
            parts = []
            for idx, g in enumerate(group):
                txt = g["w"].upper().replace("{", "").replace("}", "")
                color = "&H0000FFFF&" if idx == k else "&H00FFFFFF&"  # yellow / white
                parts.append(f"{{\\c{color}}}{txt}")
            lines.append(f"Dialogue: 0,{_ass_time(s)},{_ass_time(e)},Default,,0,0,0,,{' '.join(parts)}")
    if brand_text:
        safe = brand_text.replace("{", "").replace("}", "")
        lines.append(f"Dialogue: 1,{_ass_time(0)},{_ass_time(clip_end - clip_start)},Brand,,0,0,0,,{safe}")
    Path(path).write_text(header + "\n".join(lines), encoding="utf-8")


# ---------- 6. Cut + crop + captions ----------
def render_clip(video_path, segments, clip, out_dir, index, captions=True, face_track=True,
                logo_path=None, brand_text=""):
    out_dir = Path(out_dir)
    start, end = clip["start"], clip["end"]
    w, h, cx = (0, 0, 0)
    try:
        w, h, cx = face_center(video_path, start, end) if face_track else face_center(video_path, start, start)
        if not face_track:
            cx = w / 2
    except Exception:
        pass
    if w and h:
        cw = int(h * 9 / 16) // 2 * 2
        x = int(min(max(cx - cw / 2, 0), w - cw)) // 2 * 2
        crop = f"crop={cw}:{h}:{x}:0,"
    else:
        crop = "crop=ih*9/16:ih,"
    vf = crop + "scale=1080:1920"

    out_name = f"clip_{index}.mp4"
    ass_name = f"clip_{index}.ass"
    words = [w for s in segments for w in s["words"]] if captions else []
    if captions or brand_text:
        make_ass(words, start, end, out_dir / ass_name, brand_text=brand_text)
        vf += f",subtitles={ass_name}"   # relative path (Windows path problem se bachne ke liye)

    src = str(Path(video_path).resolve())
    dur = str(round(end - start, 2))
    enc = ["-c:v", "libx264", "-preset", "fast", "-crf", "20", "-c:a", "aac", "-b:a", "160k"]
    if logo_path and Path(logo_path).exists():
        fc = f"[0:v]{vf}[v];[1:v]scale=170:-1[lg];[v][lg]overlay=W-w-40:70[out]"
        cmd = ["ffmpeg", "-y", "-ss", str(start), "-t", dur, "-i", src,
               "-i", str(Path(logo_path).resolve()),
               "-filter_complex", fc, "-map", "[out]", "-map", "0:a?", *enc, out_name]
    else:
        cmd = ["ffmpeg", "-y", "-ss", str(start), "-i", src, "-t", dur, "-vf", vf, *enc, out_name]
    r = subprocess.run(cmd, cwd=str(out_dir), capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("ffmpeg error:\n" + r.stderr[-1500:])
    return out_dir / out_name
