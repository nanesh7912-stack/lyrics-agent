# Usage (PowerShell):  .\make_video.ps1 song.mp3 -Out out.mp4 [-Language hi] [-Title "My Song"] [-Outro outro.mp4] [-Style scene-hero]
# Add -Lyrics lyrics.txt when you want to supply the words yourself.
param(
  [Parameter(Mandatory=$true)][string]$Audio,
  [string]$Lyrics = "",
  [Parameter(Mandatory=$true)][string]$Out,
  [string]$Title = "",
  [string]$Outro = "",
  [string]$Language = "",
  [switch]$Romanize,
  [string]$Style = "scene-hero",
  [double]$Target = 0
)
$argsList = @("-m","ai_lyrics_hero","make","--audio",$Audio,"--out",$Out,"--style",$Style)
if ($Lyrics) { $argsList += @("--lyrics",$Lyrics) }
if ($Language) { $argsList += @("--language",$Language) }
if ($Romanize) { $argsList += "--romanize" }
if ($Title)  { $argsList += @("--title",$Title) }
if ($Outro)  { $argsList += @("--outro",$Outro) }
if ($Target -gt 0) { $argsList += @("--target-duration",$Target) }
python @argsList
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$base = [System.IO.Path]::ChangeExtension($Out, $null)
if (Test-Path "${base}.auto.lrc") { Write-Host "Auto-created lyrics: ${base}.auto.lrc (review spelling before publishing)." }
Write-Host "Done. Open the .review.txt next to the video for timing confidence, uncertain words, scenes and images."
