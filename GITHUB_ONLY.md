# GitHub-only video rendering

This workflow runs the project's Python/FFmpeg renderer on a GitHub-hosted runner. It does not call the Render.com service. Start jobs and download their results from the GitHub Actions page, including from a phone browser.

## Privacy comes first

The `lyrics-agent` repository is currently public. Song files committed to a public repository can be downloaded by anyone. The workflow checks the repository visibility and refuses to render unless the repository is private.

Before uploading audio:

1. In GitHub, open this repository's **Settings**.
2. Open **General** and use **Change repository visibility** to make it private.
3. Upload the audio file to the `incoming/` folder. Browser uploads are limited to 25 MiB per file.
4. The uploaded audio is stored in the private repository's Git history. Deleting the file later removes it from the latest version, but not from earlier commits.

I did not change the repository's visibility. It is still public until you change it in GitHub.

## Render from a phone

After this workflow is merged into `main` and the repository is private:

1. Sign in to GitHub in your phone browser and open this repository.
2. Upload your song into `incoming/` using **Add file → Upload files**, then commit the upload.
3. Open **Actions → Render lyric video on GitHub → Run workflow**.
4. Enter the file path, such as `incoming/song.mp3`. Optionally enter a reviewed `.txt` or `.lrc` file under `incoming/`.
5. Choose the language, model, style, resolution, and duration, then start the run.
6. When it succeeds, download the `lyric-video-…` artifact from that workflow run. It contains the MP4 plus the generated lyric and timing review files.

The workflow offers 720p, 1080p, and 4K output. 1080p is the default. Large 4K jobs can take much longer on GitHub's standard CPU runner.

## Lyrics and outside network requests

The workflow defaults to automatic speech recognition and keeps the LRCLIB lookup off. You can switch on the lyric catalog option in the run form; that contacts LRCLIB. Automatic speech recognition produces a draft and cannot guarantee every sung word's spelling, so review the included `.auto.lrc` and review report before publishing. You can upload a corrected `.lrc` file for a later render.

GitHub Actions still downloads the Python packages from PyPI and, on a cache miss, the speech model from Hugging Face. Those are software/model downloads, not app hosting. If by “GitHub only” you mean no requests to any non-GitHub domain at all, this workflow does not meet that stricter requirement yet.

The generated files are stored as a GitHub Actions artifact for 30 days. This workflow does not commit the rendered video back into the repository.
