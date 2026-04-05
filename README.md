# YouTube Transcript Search

Search any YouTube channel's subtitles for mentions of a person. Enter a channel URL and a name — the tool downloads transcripts, expands the name into variants and misspellings, searches all of them, and optionally uses the Anthropic API to confirm ambiguous single-word hits.

## Install

```bash
pip install -r requirements.txt
```

## Setup

```bash
cp .env.example .env
# edit .env and add your ANTHROPIC_API_KEY
```

LLM verification is optional. If no API key is set, ambiguous single-word matches are still returned but marked as unverified.

## Run

```bash
uvicorn main:app --reload
```

## Open

```
http://localhost:8000
```

## Notes

- Transcripts are cached in `transcripts/` — re-runs won't re-download unless files are deleted.
- Results are saved as JSON in `results/` after each search.
- Both `transcripts/` and `results/` are gitignored. Treat all search terms and aliases as local-only.
- Large channels can take 10–30 minutes to download. The UI polls for progress and won't time out.
