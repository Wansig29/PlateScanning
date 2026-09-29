# Create a Desktop shortcut to the built app, so it can be launched like any other program.
# Run this on whichever computer will use the app, after copying dist\PlateScanner there
# (see README.md > Packaging).
#   .\make_shortcut.ps1
#   .\make_shortcut.ps1 -ExePath "D:\PlateScanner\PlateScanner.exe"
param(
    [string]$ExePath = "dist\PlateScanner\PlateScanner.exe",
    [string]$Name = "PSAU Gate Plate Scanner"
)
$ErrorActionPreference = "Stop"

$exe = (Resolve-Path $ExePath).Path
$desktop = [Environment]::GetFolderPath("Desktop")
$linkPath = Join-Path $desktop "$Name.lnk"

$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($linkPath)
$shortcut.TargetPath = $exe
$shortcut.WorkingDirectory = Split-Path $exe -Parent
$shortcut.IconLocation = $exe
$shortcut.Description = "PSAU Gate Plate Scanner"
$shortcut.Save()

Write-Host "Shortcut created: $linkPath -> $exe"
