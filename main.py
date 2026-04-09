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
import base64

from fastapi import BackgroundTasks, FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

# --- Configuration ---
TRANSCRIPTS_DIR = "./transcripts"
RESULTS_DIR = "./results"
LLM_MODEL = "gpt-5.4-mini"
LLM_ENABLED = True

app = FastAPI()

# Session cookies — seeded from env var, overridable via /upload-cookies
session_cookies_b64: Optional[str] = os.getenv("YOUTUBE_COOKIES") or None

status: dict = {
    "stage": "idle",
    "videos_done": 0,
    "videos_total": 0,
    "message": "Ready",
    "log_lines": [],
    "log_total": 0,
    "results": [],
    "total_matches": 0,
    "total_videos_searched": 0,
    "total_videos_processed": 0,
    "terms": [],
    "report": None,
}

current_proc: Optional[subprocess.Popen] = None
cancel_requested: bool = False
search_running: bool = False
search_generation: int = 0


# --- Name expansion ---

def expand_name(name: str) -> list[str]:
    # Strip dots so "R." → "r", "O." → "o"
    parts = [p.lower().strip(".") for p in name.strip().split() if p.strip(".")]
    variants = set()

    if not parts:
        return []

    if len(parts) == 1:
        variants.add(parts[0])

    elif len(parts) == 2:
        first, last = parts
        variants.update([last, first, f"{first} {last}"])
        if len(first) > 1:
            variants.update([f"{first[0]} {last}", f"{first[0]}{last}"])

    else:  # 3+ parts
        first, last = parts[0], parts[-1]
        mid = parts[1]  # primary middle name/initial

        variants.update([
            " ".join(parts),          # rasim ozan atmar
            f"{first} {last}",        # rasim atmar
            f"{mid} {last}",          # ozan atmar
            last,                     # atmar
            first,                    # rasim
            mid,                      # ozan
        ])

        fi, mi = first[0], mid[0]

        # First initial + last
        if len(first) > 1:
            variants.update([f"{fi} {last}", f"{fi}{last}"])

        # Combined initials + last (ro atmar, roatmar, r o atmar)
        variants.update([
            f"{fi}{mi} {last}",       # ro atmar
            f"{fi}{mi}{last}",        # roatmar
            f"{fi} {mi} {last}",      # r o atmar
            f"{fi} {mid} {last}",     # r ozan atmar
            f"{first} {mi} {last}",   # rasim o atmar
        ])

        # Any extra middle parts (4+ word names)
        for extra in parts[2:-1]:
            variants.update([extra, f"{extra} {last}", f"{first} {extra} {last}"])

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


# --- LLM ---

def generate_report(matches: list[dict], target_name: str, extra_terms: Optional[list[str]] = None) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "No API key configured — report unavailable."
    try:
        from openai import OpenAI

        extra_terms = [t.lower().strip() for t in (extra_terms or []) if t.strip()]

        # Separate former names from project/topic terms.
        # Heuristic: single common first names are likely former names; everything else is a project.
        COMMON_FIRST_NAMES = {"justin", "justing", "alex", "chris", "pat", "sam", "jordan"}
        former_names = [t for t in extra_terms if t in COMMON_FIRST_NAMES]
        project_terms = [t for t in extra_terms if t not in COMMON_FIRST_NAMES]

        name_terms = expand_name(target_name)

        context_block = f"""You are helping determine whether YouTube videos discuss a specific person: {target_name}.

Key facts about this person:
- She is a trans woman (use she/her).
- Her current name is {target_name}. Transcripts may use her current name or name variants: {", ".join(sorted(name_terms))}.
- Former name before transition: {", ".join(former_names) if former_names else "not provided"}. Mentions of this name may refer to her, especially if the video is about her field of work.
- She is known for founding/maintaining these open-source ML/robotics projects and organizations: {", ".join(project_terms) if project_terms else "not provided"}.
  A mention of any of these terms in an ML or robotics context is likely referring to her work, even if her name is not said aloud.
  Some videos may show her old GitHub/Twitter account or photo while discussing these projects — that still counts.

For each match below, classify it as one of:
- CONFIRMED: Clearly about her — current or former name used, or her project mentioned with clear attribution to her as creator/maintainer.
- LIKELY: Her project/org is mentioned in a relevant ML/robotics context. She would typically be credited, even if not named in this excerpt.
- AMBIGUOUS: Could be her, worth a manual check — e.g. former name used but unclear if it's her, or a project term used without enough context.
- COINCIDENTAL: Clearly unrelated — e.g. "gym" meaning a fitness centre, "justin" as an unrelated person, generic use of a common word.

Group results by video. For CONFIRMED, LIKELY, and AMBIGUOUS matches include the timestamp URL."""

        lines = []
        for i, m in enumerate(matches[:150]):
            ts = fmttime(m["match_timestamp"]) if m.get("match_timestamp") else "?"
            url = f"https://youtube.com/watch?v={m['video_id']}&t={m['match_timestamp']}s"
            year = m["upload_date"][:4] if m.get("upload_date", "unknown") != "unknown" else "?"
            lines.append(
                f"{i+1}. \"{m['video_title']}\" ({year}) | {ts} | {url}\n"
                f"   matched term: \"{m['matched_term']}\" | transcript context: {m['context'][:250]}"
            )

        prompt = (
            context_block
            + f"\n\n---\nTotal matches: {len(matches)}"
            + (f" (showing first 150)" if len(matches) > 150 else "")
            + "\n\n"
            + "\n\n".join(lines)
            + "\n\n---\nWrite the report now. Be direct and concise. Skip COINCIDENTAL matches unless there are many — just note the count."
        )

        client = OpenAI(api_key=api_key)
        msg = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.choices[0].message.content.strip()
    except Exception as e:
        return f"Report generation failed: {e}"


def llm_verify(context: str, matched_term: str, target_name: str) -> tuple[str, str]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or not LLM_ENABLED:
        return "skipped - no API key", "unknown"
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        prompt = (
            f'In the following transcript snippet, the word "{matched_term}" appears. '
            f'Is this referring to the person named "{target_name}"? '
            f'Reply with YES or NO and a short reason.\n\nSnippet: {context}'
        )
        msg = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        reply = msg.choices[0].message.content.strip()
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

    seen_positions: set[int] = set()
    results = []

    for term in terms:
        term_lower = term.lower()
        term_len = len(term)

        pos = full_lower.find(term_lower)
        while pos != -1:
            if pos in seen_positions:
                pos = full_lower.find(term_lower, pos + 1)
                continue

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


def _vtt_snippet(vtt_path: Path, max_chars: int = 140) -> str:
    """Return first max_chars of deduplicated plain text from a VTT file."""
    try:
        text = vtt_path.read_text(encoding="utf-8", errors="ignore")
        seen: set[str] = set()
        parts: list[str] = []
        total = 0
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("WEBVTT") or "-->" in line or line.isdigit():
                continue
            clean = re.sub(r"<[^>]+>", "", line).strip()
            if clean and clean not in seen:
                seen.add(clean)
                parts.append(clean)
                total += len(clean) + 1
                if total >= max_chars:
                    break
        combined = " ".join(parts)
        if len(combined) > max_chars:
            combined = combined[:max_chars].rsplit(" ", 1)[0] + "…"
        return combined
    except Exception:
        return ""


def _vtt_title(vtt_path: Path) -> str:
    stem = re.sub(r"\.en$", "", vtt_path.stem)
    m = FNAME_RE.match(stem)
    return m.group(3).replace("_", " ") if m else stem


def _flush_vtt(vtt_path: Path, terms: list[str], target_name: str, item_number: int = 0) -> list[dict]:
    """Search a completed VTT file, push results into status, return matches."""
    matches = search_vtt(vtt_path, terms, target_name)
    status["total_videos_searched"] += 1
    status["total_videos_processed"] += 1
    if matches:
        for m in matches:
            m["item_number"] = item_number
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
    generation: int = 0,
):
    global current_proc

    target_url = videos_tab_url(channel_url)
    batch_file = None

    # --- Shared helpers (used in both cache and yt-dlp phases) ---

    def log(msg: str):
        status["log_lines"].append(msg)
        status["log_total"] += 1
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
        if status["log_lines"] and status["log_lines"][-1].startswith("\u29d7"):
            status["log_lines"][-1] = new_line
        else:
            log(new_line)

    def emit_vtt_result(vtt_path: Path, vid_id: Optional[str], item_n: int, item_total: int):
        matches = _flush_vtt(vtt_path, terms, target_name, item_n)
        title = _vtt_title(vtt_path)
        p = prefix(item_n, item_total)
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
        snippet = _vtt_snippet(vtt_path)
        if snippet:
            log(f"\u00bb {snippet}")
        return matches

    # --- Phase 1: search already-cached VTT files ---

    out_dir.mkdir(parents=True, exist_ok=True)
    cached_vtts = sorted(out_dir.glob("*.en.vtt"))
    cached_ids: set[str] = set()
    for vtt in cached_vtts:
        stem = re.sub(r"\.en$", "", vtt.stem)
        fm = FNAME_RE.match(stem)
        if fm:
            cached_ids.add(fm.group(2))

    if cached_vtts:
        n_cached = len(cached_vtts)
        status.update({
            "stage": "downloading", "videos_done": 0, "videos_total": n_cached,
            "message": f"Searching {n_cached} cached transcripts...",
        })
        for i, vtt in enumerate(cached_vtts, 1):
            if cancel_requested or search_generation != generation:
                return
            stem = re.sub(r"\.en$", "", vtt.stem)
            fm = FNAME_RE.match(stem)
            vid_id = fm.group(2) if fm else None
            title = _vtt_title(vtt)
            log(f"\u29d7({i}/{n_cached}) [{title}]")
            emit_vtt_result(vtt, vid_id, i, n_cached)
            status.update({
                "videos_done": i,
                "message": f"Searching cached transcripts... ({i}/{n_cached})",
            })

    # --- Phase 2: run yt-dlp for any remaining/new videos ---

    if cached_vtts:
        log(f"Cache: {len(cached_ids)} video(s) searched. Checking for new videos...")

    # Write cookies file if available
    cookies_file = None
    if session_cookies_b64:
        cookies_file = tempfile.NamedTemporaryFile(
            mode="wb", suffix=".txt", delete=False
        )
        cookies_file.write(base64.b64decode(session_cookies_b64))
        cookies_file.close()

    if sample_size is not None and random_sample:
        all_ids = fetch_video_ids(target_url)
        if not all_ids:
            log("Could not fetch video list from YouTube.")
            return
        chosen = random.sample(all_ids, min(sample_size, len(all_ids)))
        missing = [v for v in chosen if v not in cached_ids]
        if not missing:
            log("All sampled videos already cached.")
            return
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("\n".join(f"https://www.youtube.com/watch?v={v}" for v in missing))
            batch_file = f.name
        extra_args = ["--batch-file", batch_file]
        status.update({"videos_done": 0, "videos_total": len(missing),
                        "message": f"Downloading {len(missing)} sampled transcripts..."})
    elif sample_size is not None:
        extra_args = ["--playlist-end", str(sample_size), target_url]
        status.update({"videos_done": 0, "videos_total": sample_size,
                        "message": f"Downloading first {sample_size} transcripts..."})
    else:
        extra_args = [target_url]
        status.update({"message": "Downloading transcripts..."})

    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--skip-download", "--write-sub", "--write-auto-sub",
        "--sub-lang", "en", "--convert-subs", "vtt",
        "--ignore-no-formats-error", "--no-overwrites",
        "--sleep-interval", "2", "--max-sleep-interval", "5",
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
    consecutive_no_sub = 0
    cookie_warning_shown = False

    def flush_and_log():
        nonlocal current_vtt, has_transcript
        if current_vtt:
            emit_vtt_result(current_vtt, current_video_id, current_item, current_total)
            current_vtt = None
        elif has_transcript is False and (current_item > 0 or current_video_id):
            if current_video_id and current_video_id in cached_ids:
                # Already searched in Phase 1 — skip silently
                pass
            else:
                # Not cached — check disk in case yt-dlp was rate-limited but file exists
                fallback = (
                    next(out_dir.glob(f"*_{current_video_id}_*.en.vtt"), None)
                    if current_video_id else None
                )
                if fallback:
                    cached_ids.add(current_video_id)
                    emit_vtt_result(fallback, current_video_id, current_item, current_total)
                else:
                    url = yt_url(current_video_id)
                    p = prefix(current_item, current_total)
                    log_or_replace(f"{p}no transcript  {url}")
                    status["total_videos_processed"] += 1
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
            new_id = m.group(1)
            if new_id != current_video_id:
                current_video_id = new_id
                if current_total == 0:  # batch/single mode — log stub so user sees activity
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
            # Skip files already searched in Phase 1
            stem = re.sub(r"\.en$", "", vtt_path.stem)
            fm = FNAME_RE.match(stem)
            vid_id = fm.group(2) if fm else None
            if vid_id and vid_id in cached_ids:
                has_transcript = True  # prevents spurious "no transcript" log
                current_vtt = None
                consecutive_no_sub = 0
                continue
            if current_vtt != vtt_path:
                current_vtt = vtt_path
                has_transcript = True
                consecutive_no_sub = 0
                title = _vtt_title(current_vtt)
                p = prefix(current_item, current_total)
                log(f"\u29d7{p}[{title}]")  # stub line; replaced when result known
            continue

        # 100% done — flush immediately (fast path)
        if current_vtt and DONE_LINE.search(line):
            emit_vtt_result(current_vtt, current_video_id, current_item, current_total)
            current_vtt = None
            continue

        # Surface [info] lines that explain why subtitles are unavailable
        if line.startswith("[info]") and ("subtitle" in line.lower() or "caption" in line.lower()):
            log(f"yt-dlp: {line[7:].strip()}")
            if "no subtitles" in line.lower() or "no supported" in line.lower():
                consecutive_no_sub += 1
                if consecutive_no_sub == 5 and not cookie_warning_shown and session_cookies_b64:
                    cookie_warning_shown = True
                    log("⚠ YouTube is returning no subtitles for every video — your cookies may have expired. Use 'Connect to YouTube' to upload fresh cookies.")
            continue

        # Stop on any yt-dlp error
        if line.startswith("ERROR:"):
            log(f"error: {line[7:].strip()}")
            status.update({"stage": "error", "message": line[7:].strip()})
            proc.kill()
            break

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
    sample_size: Optional[int] = None, random_sample: bool = False,
    generation: int = 0,
):
    global cancel_requested, search_running
    cancel_requested = False
    search_running = True
    try:
        slug = channel_slug(channel_url)
        out_dir = Path(TRANSCRIPTS_DIR) / slug

        terms = list(dict.fromkeys(
            expand_name(name) + [t.lower() for t in extra_terms if t.strip()]
        ))
        status["terms"] = sorted(terms)

        stream_download_and_search(channel_url, out_dir, terms, name, sample_size, random_sample, generation)

        if status.get("stage") == "error":
            return
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

        if m > 0 and os.getenv("ANTHROPIC_API_KEY"):
            status["log_lines"].append("Generating AI report...")
            status["report"] = "generating"
            status["report"] = generate_report(status["results"], name, extra_terms)
            status["log_lines"][-1] = "AI report ready."

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


@app.post("/preview-terms")
async def preview_terms(req: SearchRequest):
    terms = list(dict.fromkeys(
        expand_name(req.name) + [t.lower() for t in req.extra_terms if t.strip()]
    ))
    return {"terms": sorted(terms)}


@app.post("/search")
async def search(req: SearchRequest, background_tasks: BackgroundTasks):
    global cancel_requested, search_running, search_generation
    search_generation += 1
    gen = search_generation

    if search_running:
        cancel_requested = True
        search_running = False
        if current_proc and current_proc.poll() is None:
            current_proc.kill()

    status.update({
        "stage": "starting",
        "videos_done": 0, "videos_total": 0,
        "message": "Initializing...",
        "log_lines": [], "log_total": 0, "results": [], "terms": [], "report": None,
        "total_matches": 0, "total_videos_searched": 0, "total_videos_processed": 0,
        "result": None,
    })

    background_tasks.add_task(
        run_search, req.channel_url, req.name, req.extra_terms,
        req.sample_size, req.random_sample, gen
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


@app.get("/cookie-status")
async def cookie_status():
    return {"connected": session_cookies_b64 is not None}


@app.post("/upload-cookies")
async def upload_cookies(file: UploadFile = File(...)):
    global session_cookies_b64
    content = await file.read()
    session_cookies_b64 = base64.b64encode(content).decode()
    return {"status": "ok"}


@app.get("/status")
async def get_status():
    return JSONResponse(status)


@app.get("/")
async def root():
    return FileResponse("frontend/index.html")


app.mount("/static", StaticFiles(directory="frontend"), name="static")
