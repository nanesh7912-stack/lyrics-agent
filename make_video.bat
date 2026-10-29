@echo off
REM Usage: make_video.bat song.mp3 lyrics.txt output.mp4 "Title"
if "%~3"=="" (echo Usage: make_video.bat song.mp3 lyrics.txt output.mp4 "Title" & exit /b 1)
python -m ai_lyrics_hero make --audio "%~1" --lyrics "%~2" --out "%~3" --title "%~4" --style scene-hero
if errorlevel 1 exit /b 1
echo Done. Open the .review.txt file next to the video.
