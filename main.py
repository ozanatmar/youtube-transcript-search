import json
import os
import random
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

# --- Configuration ---
TRANSCRIPTS_DIR = "./transcripts"
RESULTS_DIR = "./results"
LLM_MODEL = "claude-sonnet-4-20250514"
LLM_ENABLED = True

app = FastAPI()

status: dict = {
    "stage": "idle",
    "videos_done": 0,
    "videos_total": 0,
    "message": "Ready",
    "log_lines": [],
    "results": [],
    "total_matches": 0,
    "total_videos_searched": 0,
}

current_proc: Optional[subprocess.Popen] = None
cancel_requested: bool = False
search_running: bool = False


# --- Name expansion ---

def misspellings(word: str) -> list[str]:
    vowels = "aeiou"
    variants = set()
    for i, ch in enumerate(word):
        if ch in vowels:
            variants.add(word[:i] + word[i+1:])
    for i, ch in enumerate(word[:-1]):
        if ch in vowels:
            for v in [v for v in vowels if v != ch][:2]:
                variants.add(word[:i] + v + word[i+1:])
    return [v for v in variants if v and v != word]


def expand_name(name: str) -> list[str]:
    parts = name.strip().split()
    variants = set()
    if len(parts) == 1:
        variants.add(parts[0].lower())
    elif len(parts) >= 2:
        first, last = parts[0].lower(), parts[-1].lower()
        variants.update([
            f"{first} {last}", f"{first}{last}",
            f"{first[0]}{last}", f"{first}{last[0]}",
            last, first,
        ])
        for m in misspellings(first):
            variants.update([m, f"{m} {last}", f"{m}{last}"])
        for m in misspellings(last):
            variants.update([m, f"{first} {m}", f"{first}{m}"])
    return [v for v in variants if len(v) > 1]


# --- VTT cleaning ---

def _ts_to_secs(ts: str) -> float:
    ts = ts.replace(",", ".")
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def parse_vtt_cues(raw: str) -> list[tuple[float, str]]:
    """Parse VTT into (start_secs, text) pairs, deduplicating rolling captions."""
    ts_re = re.compile(r"^(\d{2}:\d{2}:\d{2}[.,]\d+)\s*-->")
    cues: list[tuple[float, list[str]]] = []
    cur_ts = 0.0
    cur_lines: list[str] = []
    in_cue = False

    for line in raw.splitlines():
        line = line.strip()
        m = ts_re.match(line)
        if m:
            if in_cue and cur_lines:
                cues.append((cur_ts, cur_lines))
            cur_ts = _ts_to_secs(m.group(1))
            cur_lines = []
            in_cue = True
            continue
        if not line:
            if in_cue and cur_lines:
                cues.append((cur_ts, cur_lines))
            in_cue = False
            cur_lines = []
            continue
        if not in_cue:
            continue
        if line.startswith(("WEBVTT", "Kind:", "Language:")) or re.match(r"^\d+$", line):
            continue
        line = re.sub(r"<\d{2}:\d{2}:\d{2}[.,]\d+>", "", line)
        line = re.sub(r"<[^>]+>", "", line).strip()
        if line:
            cur_lines.append(line)

    if in_cue and cur_lines:
        cues.append((cur_ts, cur_lines))

    # Deduplicate: YouTube rolling captions repeat previous lines.
    # For each cue, keep only words not already in the previous cue.
    result: list[tuple[float, str]] = []
    prev_words: list[str] = []
    for ts, lines in cues:
        words = " ".join(lines).split()
        # Find longest prefix of `words` that matches a suffix of `prev_words`
        new_start = 0
        for overlap in range(min(len(prev_words), len(words)), 0, -1):
            if prev_words[-overlap:] == words[:overlap]:
                new_start = overlap
                break
        new_words = words[new_start:]
        if new_words:
            result.append((ts, " ".join(new_words)))
        prev_words = words

    return result


def extract_context(text: str, match_start: int, match_end: int, window: int = 200) -> str:
    start = max(0, match_start - window)
    end = min(len(text), match_end + window)
    snippet = text[start:end]
    if start > 0:
        snippet = "..." + snippet
    if end < len(text):
        snippet += "..."
    return snippet


# --- LLM verification ---

def llm_verify(context: str, matched_term: str, target_name: str) -> tuple[str, str]:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key or not LLM_ENABLED:
        return "skipped - no API key", "unknown"
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        prompt = (
            f'In the following transcript snippet, the word "{matched_term}" appears. '
            f'Is this referring to the person named "{target_name}"? '
            f'Reply with YES or NO and a short reason.\n\nSnippet: {context}'
        )
        message = client.messages.create(
            model=LLM_MODEL, max_tokens=150,
            messages=[{"role": "user", "content": prompt}],
        )
        reply = message.content[0].text.strip()
        return reply, "high" if reply.upper().startswith("YES") else "low"
    except Exception as e:
        return f"skipped - error: {e}", "unknown"


# --- Helpers ---

def channel_slug(channel_url: str) -> str:
    slug = re.sub(r"https?://(www\.)?youtube\.com/", "", channel_url)
    slug = re.sub(r"[^\w\-]", "_", slug)
    return slug.strip("_") or "channel"


def is_video_url(url: str) -> bool:
    return "watch?v=" in url or "youtu.be/" in url


def videos_tab_url(channel_url: str) -> str:
    if is_video_url(channel_url):
        return channel_url
    url = re.sub(r"/(videos|shorts|streams|live)$", "", channel_url.rstrip("/"))
    return url + "/videos"


def fetch_video_ids(channel_url: str) -> list[str]:
    status.update({"stage": "downloading", "message": "Fetching video list..."})
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--flat-playlist", "--dump-single-json",
        "--no-warnings", "--ignore-errors",
        channel_url,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    try:
        data = json.loads(proc.stdout)
        return [e["id"] for e in (data.get("entries") or []) if e and e.get("id")]
    except Exception:
        return []


def search_vtt(vtt_path: Path, terms: list[str], target_name: str) -> list[dict]:
    """Search a single VTT file and return matches with timestamps."""
    try:
        raw = vtt_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []

    cues = parse_vtt_cues(raw)
    if not cues:
        return []

    # Build flat text with a char→timestamp map
    full_text = ""
    cue_map: list[tuple[int, int, float]] = []  # (start_char, end_char, secs)
    for ts, text in cues:
        s = len(full_text)
        full_text += text + " "
        cue_map.append((s, len(full_text), ts))

    full_lower = full_text.lower()

    stem = re.sub(r"\.en$", "", vtt_path.stem)
    fm = FNAME_RE.match(stem)
    upload_date = fm.group(1) if fm else "unknown"
    video_id    = fm.group(2) if fm else stem
    video_title = fm.group(3).replace("_", " ") if fm else stem

    single_word_terms = {t for t in terms if " " not in t}
    seen_positions: set[int] = set()
    results = []

    for term in terms:
        term_lower = term.lower()
        term_len = len(term)
        llm_verdict, confidence = None, None
        llm_checked = False

        pos = full_lower.find(term_lower)
        while pos != -1:
            if pos in seen_positions:
                pos = full_lower.find(term_lower, pos + 1)
                continue

            # LLM verification on first occurrence only
            if not llm_checked and term in single_word_terms and len(target_name.split()) >= 2:
                context = extract_context(full_text, pos, pos + term_len)
                llm_verdict, confidence = llm_verify(context, term, target_name)
                llm_checked = True
                if llm_verdict.upper().startswith("NO"):
                    break  # skip all occurrences of this term

            match_ts = 0
            for s, e, ts in cue_map:
                if s <= pos < e:
                    match_ts = int(ts)
                    break

            seen_positions.add(pos)
            context = extract_context(full_text, pos, pos + term_len)
            results.append({
                "video_id": video_id,
                "video_title": video_title,
                "upload_date": upload_date,
                "match_timestamp": match_ts,
                "matched_term": term,
                "context": context,
                "llm_verdict": llm_verdict,
                "confidence": confidence,
            })

            pos = full_lower.find(term_lower, pos + term_len)

    return results


# --- Combined download + search stream ---

DETAIL_LINE   = re.compile(r"^\[download\]\s+[\d.]+\s*(KiB|MiB|GiB|B)\s")
ITEM_LINE     = re.compile(r"\[download\] Downloading item (\d+) of (\d+)")
WRITE_LINE    = re.compile(r"\[info\] Writing video subtitles to: (.+\.vtt)")
DEST_LINE     = re.compile(r"\[download\] Destination: (.+\.vtt)")
ALREADY_LINE  = re.compile(r"\[download\] (.+\.vtt) has already been downloaded")
DONE_LINE     = re.compile(r"\[download\] 100%")
VIDEO_ID_LINE = re.compile(r"\[youtube\] ([A-Za-z0-9_\-]+): Downloading webpage")
EXTRACT_LINE  = re.compile(r"\[youtube\] Extracting URL:.*[?&]v=([A-Za-z0-9_\-]{11})")

# Filename pattern: YYYYMMDD_<11-char video ID>_<title>
FNAME_RE = re.compile(r"^(\d{8})_([A-Za-z0-9_\-]{11})_(.+)$")


def fmttime(secs: int) -> str:
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _vtt_title(vtt_path: Path) -> str:
    stem = re.sub(r"\.en$", "", vtt_path.stem)
    m = FNAME_RE.match(stem)
    return m.group(3).replace("_", " ") if m else stem


def _flush_vtt(vtt_path: Path, terms: list[str], target_name: str) -> list[dict]:
    """Search a completed VTT file, push results into status, return matches."""
    matches = search_vtt(vtt_path, terms, target_name)
    status["total_videos_searched"] += 1
    if matches:
        status["results"].extend(matches)
        status["total_matches"] = len(status["results"])
    return matches


def stream_download_and_search(
    channel_url: str,
    out_dir: Path,
    terms: list[str],
    target_name: str,
    sample_size: Optional[int],
    random_sample: bool,
):
    global current_proc

    target_url = videos_tab_url(channel_url)
    batch_file = None

    if sample_size is not None and random_sample:
        all_ids = fetch_video_ids(target_url)
        if not all_ids:
            return
        chosen = random.sample(all_ids, min(sample_size, len(all_ids)))
        status.update({
            "stage": "downloading", "videos_done": 0, "videos_total": len(chosen),
            "message": f"Sampled {len(chosen)} of {len(all_ids)} randomly. Downloading...",
        })
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("\n".join(f"https://www.youtube.com/watch?v={vid}" for vid in chosen))
            batch_file = f.name
        extra_args = ["--batch-file", batch_file]
    elif sample_size is not None:
        status.update({
            "stage": "downloading", "videos_done": 0, "videos_total": sample_size,
            "message": f"Downloading first {sample_size} videos...",
        })
        extra_args = ["--playlist-end", str(sample_size), target_url]
    else:
        status.update({"stage": "downloading", "message": "Starting download..."})
        extra_args = [target_url]

    # Write cookies file if YOUTUBE_COOKIES env var is set
    cookies_file = None
    cookies_b64 = os.getenv("YOUTUBE_COOKIES")
    if cookies_b64:
        import base64, tempfile
        cookies_file = tempfile.NamedTemporaryFile(
            mode="wb", suffix=".txt", delete=False
        )
        cookies_file.write(base64.b64decode(cookies_b64))
        cookies_file.close()

    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--skip-download", "--write-sub", "--write-auto-sub",
        "--sub-lang", "en", "--convert-subs", "vtt",
        "--ignore-no-formats-error",
        "--output", str(out_dir / "%(upload_date)s_%(id)s_%(title)s.%(ext)s"),
        "--no-warnings", "--ignore-errors",
    ]
    if cookies_file:
        cmd += ["--cookies", cookies_file.name]
    cmd += extra_args

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
    )
    current_proc = proc

    current_vtt: Optional[Path] = None
    current_video_id: Optional[str] = None
    current_item = 0
    current_total = 0
    has_transcript = False
    playlist_logged = False

    def log(msg: str):
        status["log_lines"].append(msg)
        if len(status["log_lines"]) > 500:
            status["log_lines"] = status["log_lines"][-500:]

    def yt_url(vid_id: Optional[str], ts: int = 0) -> str:
        if not vid_id:
            return ""
        url = f"https://youtube.com/watch?v={vid_id}"
        if ts:
            url += f"&t={ts}s"
        return url

    def prefix(n, total):
        return f"({n}/{total}) " if total else ""

    def log_or_replace(new_line: str):
        """Replace last log line if it's a pending 'loading' stub, else append."""
        if status["log_lines"] and status["log_lines"][-1].startswith("\u29d7"):
            status["log_lines"][-1] = new_line
        else:
            log(new_line)

    def flush_and_log():
        nonlocal current_vtt, has_transcript
        if current_vtt:
            vid_id = current_video_id
            matches = _flush_vtt(current_vtt, terms, target_name)
            title = _vtt_title(current_vtt)
            n = current_item
            p = prefix(n, current_total)
            if matches:
                unique_terms = list(dict.fromkeys(m["matched_term"] for m in matches))
                terms_hit = ", ".join(unique_terms)
                ts = matches[0].get("match_timestamp", 0)
                ts_str = f" @ {fmttime(ts)}" if ts else ""
                count_str = f" ({len(matches)}×)" if len(matches) > 1 else ""
                url = yt_url(vid_id, ts)
                log_or_replace(f"{p}[{title}] — found: {terms_hit}{ts_str}{count_str}  {url}")
            else:
                url = yt_url(vid_id)
                log_or_replace(f"{p}[{title}] — no match  {url}")
            current_vtt = None
        elif has_transcript is False and (current_item > 0 or current_video_id):
            url = yt_url(current_video_id)
            p = prefix(current_item, current_total)
            log_or_replace(f"{p}no transcript  {url}")
        has_transcript = False

    for raw_line in proc.stdout:
        if cancel_requested:
            proc.kill()
            break

        line = raw_line.strip()
        if not line:
            continue

        # Log playlist loading once
        if "Downloading API JSON" in line and not playlist_logged:
            log("Loading channel playlist...")
            playlist_logged = True
            continue
        if "Downloading API JSON" in line:
            continue

        # Track current video ID (two possible patterns)
        m = VIDEO_ID_LINE.search(line) or EXTRACT_LINE.search(line)
        if m:
            current_video_id = m.group(1)
            if current_total == 0:  # single video mode — log it so the user sees activity
                log(f"\u29d7[{current_video_id}]")
            continue

        # New item — flush previous
        m = ITEM_LINE.search(line)
        if m:
            flush_and_log()
            current_item, current_total = int(m.group(1)), int(m.group(2))
            has_transcript = False
            status.update({
                "stage": "downloading",
                "videos_done": current_item,
                "videos_total": current_total,
                "message": f"Downloading subtitles... ({current_item}/{current_total})",
            })
            continue

        # Detect VTT file path (new download, re-download destination, or already cached)
        m = WRITE_LINE.search(line) or DEST_LINE.search(line) or ALREADY_LINE.search(line)
        if m:
            vtt_path = Path(m.group(1))
            if current_vtt != vtt_path:  # only log stub once per file
                current_vtt = vtt_path
                has_transcript = True
                title = _vtt_title(current_vtt)
                p = prefix(current_item, current_total)
                log(f"\u29d7{p}[{title}]")  # stub line; replaced when result known
            continue

        # 100% done — flush immediately (fast path)
        if current_vtt and DONE_LINE.search(line):
            vid_id = current_video_id
            matches = _flush_vtt(current_vtt, terms, target_name)
            title = _vtt_title(current_vtt)
            n = current_item
            p = prefix(n, current_total)
            if matches:
                unique_terms = list(dict.fromkeys(mx["matched_term"] for mx in matches))
                terms_hit = ", ".join(unique_terms)
                ts = matches[0].get("match_timestamp", 0)
                ts_str = f" @ {fmttime(ts)}" if ts else ""
                count_str = f" ({len(matches)}×)" if len(matches) > 1 else ""
                url = yt_url(vid_id, ts)
                new_line = f"{p}[{title}] — found: {terms_hit}{ts_str}{count_str}  {url}"
            else:
                url = yt_url(vid_id)
                new_line = f"{p}[{title}] — no match  {url}"
            log_or_replace(new_line)
            current_vtt = None
            continue

        # Surface [info] lines that explain why subtitles are unavailable
        if line.startswith("[info]") and ("subtitle" in line.lower() or "caption" in line.lower()):
            log(f"yt-dlp: {line[7:].strip()}")

    proc.wait()
    current_proc = None
    flush_and_log()  # flush final item
    if batch_file:
        os.unlink(batch_file)
    if cookies_file:
        os.unlink(cookies_file.name)


# --- Background task ---

def run_search(
    channel_url: str, name: str, extra_terms: list[str],
    sample_size: Optional[int] = None, random_sample: bool = False
):
    global cancel_requested, search_running
    cancel_requested = False
    search_running = True
    try:
        slug = channel_slug(channel_url)
        out_dir = Path(TRANSCRIPTS_DIR) / slug
        out_dir.mkdir(parents=True, exist_ok=True)

        terms = list(dict.fromkeys(
            expand_name(name) + [t.lower() for t in extra_terms if t.strip()]
        ))

        stream_download_and_search(channel_url, out_dir, terms, name, sample_size, random_sample)

        if cancel_requested:
            status.update({"stage": "cancelled", "message": "Search cancelled."})
            return

        output = {
            "total_videos_searched": status["total_videos_searched"],
            "total_matches": status["total_matches"],
            "results": status["results"],
        }

        Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        (Path(RESULTS_DIR) / f"{slug}_{ts}.json").write_text(
            json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        m = status["total_matches"]
        v = status["total_videos_searched"]
        summary = f"Search complete — {m} match{'es' if m != 1 else ''} found across {v} video{'s' if v != 1 else ''} with transcripts."
        status["log_lines"].append(summary)
        status.update({
            "stage": "done",
            "message": f"Done. Found {m} matches across {v} videos.",
            "result": output,
        })

    except Exception as e:
        status["log_lines"].append(f"Search failed: {e}")
        status.update({"stage": "error", "message": str(e)})
    finally:
        search_running = False


# --- API ---

class SearchRequest(BaseModel):
    channel_url: str
    name: str
    extra_terms: list[str] = []
    sample_size: Optional[int] = None
    random_sample: bool = False


@app.post("/search")
async def search(req: SearchRequest, background_tasks: BackgroundTasks):
    global cancel_requested, search_running
    if search_running:
        cancel_requested = True
        search_running = False
        if current_proc and current_proc.poll() is None:
            current_proc.kill()

    status.update({
        "stage": "starting",
        "videos_done": 0, "videos_total": 0,
        "message": "Initializing...",
        "log_lines": [], "results": [],
        "total_matches": 0, "total_videos_searched": 0,
        "result": None,
    })

    background_tasks.add_task(
        run_search, req.channel_url, req.name, req.extra_terms,
        req.sample_size, req.random_sample
    )
    return {"status": "started"}


@app.post("/cancel")
async def cancel():
    global cancel_requested, search_running
    cancel_requested = True
    search_running = False
    if current_proc and current_proc.poll() is None:
        current_proc.kill()
    status.update({"stage": "cancelled", "message": "Search cancelled."})
    return {"status": "cancelled"}


@app.get("/status")
async def get_status():
    return JSONResponse(status)


@app.get("/")
async def root():
    return FileResponse("frontend/index.html")


app.mount("/static", StaticFiles(directory="frontend"), name="static")
