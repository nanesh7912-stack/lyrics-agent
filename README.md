# AI Lyrics Hero

Turns a song + its lyrics into a vertical (1080x1920, 30 fps by default) lyric video: timed ASS subtitles,
kinetic "hero word" typography, meaning-matched animated scene backgrounds, music-aware motion, an optional outro with an
explicit audio policy, and a verified MP4. Pure Python (Pillow + numpy) + FFmpeg. Works offline; AI services are optional.

## What is implemented (and how it was verified)
| Area | Implemented | Verified by |
|---|---|---|
| **Lyrics preserved exactly** | your text (Romanized Hindi, English, mixed, punctuation, case, repeats) is never rewritten; ASR is only used for *timing* | `tests/test_hindi_lyrics.py` |
| **Cross-script alignment** | Romanized lyrics <-> Devanagari ASR output via an internal transliteration used only for matching; split/merged words (`dil bar`/`dilbar`); per-word source + confidence; interpolated words flagged "uncertain", never "exact" | unit tests with simulated ASR output (faster-whisper itself was not run, see limits) |
| **Duration + outro** | `main = target - outro`; frame-exact; final video **and** audio checked with ffprobe + full decode; atomic output (no partial MP4) | `tests/test_duration.py` (26 tests incl. real renders) |
| **Outro audio policies** | `continue` song keeps playing under the outro (default) / `outro` its own audio / `silent` / `mix` crossfade | spectrum analysis of each policy's rendered audio |
| **Music analysis** | energy curve, onsets, strong impacts, tempo + beat grid (autocorrelation), quiet/mid/high sections, silent-gap detection; deterministic fallback when audio is missing/sparse | `tests/test_music.py` |
| **Kinetic typography** | presets: pop/overshoot, slide x4, rotate, wipe (mask), tracking, blur, typewriter chars, impact (+shake); animated glow; karaoke-style highlight in caption style; hero word chosen from length, emotional words, phrase position, sung duration, musical accent, repetition | `tests/test_director.py`, `test_animation.py` |
| **Visual director** | per-line plan: scene, mood, camera zoom/pan, typography, composition (centered / offset / giant background word / minimal), transition, accent color; repeated lines reuse a motif; never 3 identical atmospheric scenes in a row; deterministic for a given seed | `tests/test_director.py` |
| **Scenes (all procedural)** | moon, eye (blinks), heart+fire, rain, romance, city lights, road, sunset nature, galaxy, dream, stars, night waves, light streaks, plain sky; scenes during instrumental breaks (never invented lyrics) | frame inspection + tests |
| **Transitions** | fade, zoom, slide, light wipe, flash cut; chosen from the music (cut on impacts, slower fades when quiet) | `tests/test_director.py` |
| **Your own images** | `assets/user/backgrounds/<scene>/*.jpg\|png\|webp`, cover-crop, darkening, Ken-Burns, caching, safe loading | tests |
| **Optional AI** | OpenAI-compatible LLM suggests scene/mood only for lines the dictionary can't match; strict validation; any failure -> local analysis | tests with mocked HTTP (never run against a real API) |
| **Rendering reliability** | no pipe deadlocks, exit-code checks, stderr captured, QSV auto-detected with a real test encode and automatic software retry | tests |

Tests: `python -m pytest -q tests` (275 tests; about 1 minute on a typical PC, longer on Windows because the integration tests
launch FFmpeg ~190 times). Quick loop without the real-video renders: `python -m pytest -q tests -m "not slow"` (223 tests, ~6 s).

## Install (Windows PowerShell)
```powershell
# 1. Install Python 3.10+ (python.org) and FFmpeg (e.g. `winget install Gyan.FFmpeg`), then open a NEW PowerShell
ffmpeg -version ; ffprobe -version ; python --version

# 2. Extract and enter the project (any folder / drive; spaces are fine)
Expand-Archive -Path ".\AI_Lyrics_Hero_Professional_Upgraded_FIXED.zip" -DestinationPath "." -Force
cd ".\ai_lyrics_hero"

# 3. Virtual environment + dependencies
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope Process Bypass
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Test
python -m pytest -q tests
python -m compileall -q ai_lyrics_hero
```
Linux/macOS: the same with `source .venv/bin/activate`.

## Make a video
1. **Lyrics are optional.** By default, the app first looks for an exact title/artist/duration match in LRCLIB. It uses catalog word timings when available, or aligns catalog text to the vocals; if there is no match, it creates an editable speech-recognition draft. It saves a word-timed `song.auto.lrc` file beside the output. Hindi written in Devanagari is automatically
   romanized into Latin letters (Hindi pronunciation, not an English translation). Check spellings before
   publishing, especially for sung words. You can also provide your own `lyrics.txt` or `.lrc` file; supplied lyrics are
   used as written unless you add `--romanize`. Put `[[word]]` around a word to force it as the hero word.
   The browser app scans the full transcript by default; the `make` command does this by default when
   `--target-duration` is set without a manual range. It chooses the strongest continuous section for your video
   length by scoring timed-word density, lyric coverage, repeated phrases, and confidence;
   the review report records the selected original-song timestamps. This section choice is best effort, so review spellings and timing before publishing.
2. **Audio**: mp3/wav/m4a/... (anything FFmpeg reads).
3. Run:
```powershell
# song + outro, 40 s total (33.5 s of lyrics + 6.5 s outro), song continues under the outro
# No lyrics file is needed; use --language hi for a Hindi song (or en for English).
python -m ai_lyrics_hero make --audio ".\song.mp3" --out ".\out\song.mp4" `
    --title "My Song" --language hi --style scene-hero --outro ".\outro.mp4" --target-duration 40 --outro-audio continue

# Scan the full song and choose its strongest continuous 40-second lyric section
python -m ai_lyrics_hero make --audio ".\song.mp3" --out ".\out\song_best.mp4" `
    --language hi --romanize --style scene-hero --target-duration 40

# Supply exact track details when the filename is not "Title - Artist.mp3"
python -m ai_lyrics_hero make --audio ".\song.mp3" --track-title "Song Title" --artist "Artist Name" `
    --out ".\out\song.mp4"

# Optional: provide lyrics yourself when you want exact spelling
python -m ai_lyrics_hero make --audio ".\song.mp3" --lyrics ".\lyrics.txt" --out ".\out\song.mp4"

# Reuse a generated LRC, convert Hindi to Latin letters, and render a chosen 40 s section without transcribing again
python -m ai_lyrics_hero make --audio ".\song.mp3" --lyrics ".\out\song.auto.lrc" --romanize `
    --out ".\out\song_romanized.mp4" --start 122 --end 162

# same via the helper script
.\make_video.ps1 .\song.mp3 -Out .\out\song.mp4 -Language hi -Outro .\outro.mp4 -Target 40
```
4. Open `song.auto.lrc` to correct the automatically transcribed, word-timed draft. Open `song.review.txt` (next to the video) for the source-song section used, each lyric line's original timestamp, timing confidence, **uncertain words**, the visual plan, and verified durations. Fix `song.ass` (plain text) or the lyrics and re-render without re-aligning:
   `python -m ai_lyrics_hero render --ass song.ass --audio song.mp3 --out song.mp4 --style scene-hero`

### Commands
`make` (align + render), `align` (only the ASS + review), `render` (ASS -> MP4; the original `--ass ...` form without a subcommand still works).

### Options
| Option | Meaning |
|---|---|
| `--style classic\|hero\|scene\|scene-hero` | look presets (classic = purple geometric + hero text; scene = night scenes + captions; scene-hero = scenes + hero text) |
| `--size 1080x1920 --fps 30` | resolution / frame rate (any size; default portrait) |
| `--target-duration S` | final length **including** the outro. `main = target - outro` |
| `--outro FILE --outro-duration S` | outro clip; use only S seconds of it (default: all) |
| `--outro-audio continue\|outro\|silent\|mix` `--crossfade S` | audio under the outro (see below) |
| `--start S --end S` | preview window of the song (cannot be combined with `--target-duration`) |
| `--select-best` / `--keep-start` | force full-song selection / honor `--start`; with `--target-duration`, full-song selection is the default |
| `--typography`, `--composition`, `--transitions` | force `pop\|slide\|rotate\|wipe\|tracking\|blur\|chars\|impact`, `centered\|offset\|bg_word\|minimal`, `fade\|zoom\|slide\|wipe\|cut` (default `auto`) |
| `--beat-intensity 0..2` | beat pulses/shake (0 = off) |
| `--seed N` | reproducible visual choices (same input + seed = same video) |
| `--asset-dir DIR` | your own backgrounds (see `assets/user/README.txt`) |
| `--llm openai` | optional AI scene analysis (needs `AI_LYRICS_LLM_API_KEY`, see `.env.example`) |
| `--encoder auto\|libx264\|h264_qsv` | `auto` uses Intel QSV only if a real test encode succeeds, otherwise software x264 |
| `--vocal-start/--vocal-end`, `--no-asr`, `--language`, `--model` | timing options |
| `--romanize` | display Devanagari lyrics in Latin letters (transliteration, not translation) |
| `--no-hero --no-images --no-interludes --no-verify --allow-ambiguous` | switches |

Full list of config keys: `hero_config.example.json` (use `--config file.json`; command-line flags override it).

### How audio behaves under the outro
| `--outro-audio` | During the outro you hear |
|---|---|
| `continue` (default) | the original song carries on (never truncated or looped) |
| `outro` | the outro's own audio (song ends exactly where the lyrics part ends) |
| `silent` | silence |
| `mix` | the song fades out while the outro audio fades in over `--crossfade` seconds |
If the song is shorter than the video, the remainder is **silent** and a warning is reported; music is never repeated.
The finished file's video and audio durations are checked against the plan (video: 1.5 frames, audio: 0.08 s because AAC works in 1024-sample blocks).

### Accuracy: what is and isn't guaranteed
* **Spelling**: with a lyrics file, your text is shown exactly as written (case preserved; set `"uppercase": true` for caps).
  Automatic mode first tries an exact title/artist/duration lookup in LRCLIB and uses its word-synced lyrics when available. Catalog text is community supplied and should still be checked. If there is no exact match, Faster-Whisper creates an unverified draft; no recognizer guarantees the spelling of sung words for every song. The default model is `medium` for better transcription quality, at the cost of longer processing and a larger download. Models are cached in a writable local cache; set `AI_LYRICS_HERO_MODEL_CACHE` to choose another writable folder.
* **Timing** sources: word-synced catalog timing > speech word timestamps aligned to catalog/user lyrics > line-synced LRC with interpolated word timing > energy estimate (rough, flagged). Reports identify uncertain words and lines.
* **Pictures/scenes** use exact dictionary matches (`scenes.json`, `lexicon.json`) so a word never gets an unrelated picture; ambiguous
  words (e.g. Hindi `sun` = listen) get nothing unless `--allow-ambiguous`. Unmatched lines get an abstract atmospheric scene.
* "AI" features: local rule-based analysis is always available; the LLM layer is optional and was only tested with simulated responses.

## Folder structure
```
ai_lyrics_hero/        package: cli, config, workflow, pipeline, lyrics, asr, align, text_utils, ass_writer/parser,
                       music, director, semantic, scenes(+.json), transitions, assets, renderer, hero, caption, layout,
                       animation, glow, fonts, imagery(+lexicon.json), image_anim, audio
assets/fonts/          optional local fonts; the Docker image installs system font fallbacks
assets/user/           put your own images here (README inside)
samples/               sample ASS/LRC lyrics
tests/                 275 tests
```

## FFmpeg versions and the 7.1 duration fix
Tested with FFmpeg 6.1.1 (system) and **7.1.1** (built from source; the project was reported failing on 7.1.5).
On 7.1 the old code produced audio 0.09 s short ("audio stream is 7.914s, expected 8.000s"). Root cause (reproduced and isolated by
ablation): the filter chain `tpad/trim/setpts` lost the frame rate, so the MP4 muxer gave the last video frame no duration (video =
(N-1)/fps), and `-frames:v N` then ended the whole output at that frame, cutting the audio there too. Fixed by an explicit output frame
rate (`-r` + a closing `fps` filter) and by not using `-frames:v` (the filter graph already emits exactly N frames). The verifier now
requires the video to be within half a frame and the audio within 0.05 s (about two AAC frames: encoder priming/padding only).

## Troubleshooting
* `ffmpeg not found` -> install FFmpeg and reopen PowerShell. `output verification failed` -> the message lists the exact mismatch; nothing is written to the final path.
* `h264_qsv ... not usable` -> harmless; software encoding is used. QSV is only attempted when FFmpeg can really use it.
* Devanagari lyrics look wrong -> needs Pillow with libraqm (standard wheels include it).
* Slow renders: scene styles take about 2 s of CPU per second of video; use `--size 540x960 --fps 24` for previews.

## Known limitations
* **faster-whisper was not run** in the build environment (no model download); the alignment code is tested with simulated ASR words. First real run: check the review file.
* **Intel QSV and the OpenAI-compatible LLM path were not tested on real hardware/services.**
* Scenes are simple procedural vector art (not photographic); use your own images for cinematic backgrounds. Emoji pictures look soft when large.
* Beat detection is heuristic (it reported 150 BPM for the sample song, possibly double-time); beats are only used for subtle accents.
* The browser studio is a web interface over the same renderer. Automatic transcription still needs its model and singing transcripts should be reviewed.
* Hero-word selection is rule-based, not a language model; force words with `[[word]]`.

## Browser app

AI Lyrics Hero includes a responsive browser studio. It uploads a song to the Python service and looks up lyrics using only the title, artist, and duration (never the audio file). It uses word-synced catalog timing when available; otherwise a larger speech model aligns lyrics to the vocals, and unmatched songs receive an unverified draft. Projects and render history are stored in SQLite and scoped to a private browser session. Audio and output files are stored on the service host.

### Run locally

```powershell
python -m pip install -r requirements.txt
python -m ai_lyrics_hero.webapp
```

Open <http://127.0.0.1:8000>. The service needs FFmpeg and FFprobe on its host. On first automatic transcription, Faster-Whisper downloads its model to the configured cache. For a phone or another computer on the same network, run `python -m ai_lyrics_hero.webapp --host 0.0.0.0` and open the computer's local network address in the other device's browser.

### API overview

| Endpoint | Purpose |
|---|---|
| `GET /api/session` | Create/read the browser session and CSRF token |
| `GET, POST /api/projects` | List and save this session's project settings |
| `GET, PUT, DELETE /api/projects/<id>` | Read, update, or delete a saved project |
| `GET, POST /api/jobs` | List recent renders or upload audio and start a video job |
| `GET /api/jobs/<id>` | Read render progress and status |
| `GET /api/jobs/<id>/download` | Download or preview the completed MP4 |
| `GET /api/jobs/<id>/review` | Download the timing and render review |
| `GET /api/jobs/<id>/lyrics` | Download generated word-timed LRC lyrics, when available |
| `DELETE /api/jobs/<id>` | Delete a completed or failed job and its files |

API writes require the `X-CSRF-Token` returned by `/api/session`. Jobs, projects, and downloads are restricted to the browser session that created them. The app accepts one active render at a time and audio files up to 100 MB.

### Deploy the full app

GitHub Pages can host static files, but it cannot run Python, FFmpeg, transcription, or the SQLite API. This repository includes a Render Blueprint at `ai_lyrics_hero/render.yaml` for the full app. In Render, connect this repository as a Blueprint and select that file as the Blueprint path. Review the service plan before creating it. The configuration uses a 4 GB web service and a 5 GB persistent disk because the app stores projects and model files and runs video renders. These resources are paid; create the service only if the displayed cost is acceptable to you. The Blueprint generates a private session key and enables secure cookies automatically. After deployment, check `/api/health` and use the Render service URL on a phone, tablet, or computer.

The contact page opens a prefilled email to `nanesh7912@gmail.com` in the visitor's email app. The visitor still needs an email app or webmail handler configured.

### Testing the browser app

Run `python -m pytest -q tests/test_webapp.py` for API, session, project, and job checks. Manually open the app at desktop and phone widths, test keyboard navigation and upload, create a short video, and verify the MP4, timing report, and automatic lyric draft. Use a browser device emulator for narrow screens, then test on a real phone and another browser before publishing.
