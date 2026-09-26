# Build the Windows app into dist\PlateScanner\ (run from the project root).
$ErrorActionPreference = "Stop"
$py = ".\.venv\Scripts\python.exe"
# PYTHONIOENCODING: EasyOCR's download progress bar crashes on the cp1252 console otherwise.
if (-not (Test-Path "models")) { $env:PYTHONIOENCODING = "utf-8"; & $py tools\fetch_models.py }
& $py -m pytest -q
& $py -m PyInstaller --noconfirm platescanner.spec
Write-Host "Built dist\PlateScanner\PlateScanner.exe"
