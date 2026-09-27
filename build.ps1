# Build the Windows app into dist\PlateScanner\ (run from the project root).
#   .\build.ps1                          -> dist\PlateScanner\PlateScanner.exe
#   .\build.ps1 -Dest "$env:LOCALAPPDATA\Programs"   -> install/update the local copy
# Building outside OneDrive-synced folders avoids "Access is denied" while
# OneDrive is uploading the previous build.
param([string]$Dest = "dist")
$ErrorActionPreference = "Stop"
$py = ".\.venv\Scripts\python.exe"

function Invoke-Step([string]$what, [scriptblock]$cmd) {
    & $cmd
    if ($LASTEXITCODE -ne 0) { throw "$what failed (exit code $LASTEXITCODE)" }
}

# PYTHONIOENCODING: the download progress bars crash on the cp1252 console otherwise.
if (-not (Test-Path "models\alpr")) {
    $env:PYTHONIOENCODING = "utf-8"
    Invoke-Step "Downloading models" { & $py tools\fetch_models.py }
}
Invoke-Step "Tests" { & $py -m pytest -q }
$work = Join-Path $env:LOCALAPPDATA "PlateScanner-build"
Invoke-Step "PyInstaller" { & $py -m PyInstaller --noconfirm --distpath $Dest --workpath $work platescanner.spec }
Write-Host "Built $(Join-Path $Dest 'PlateScanner\PlateScanner.exe')"
