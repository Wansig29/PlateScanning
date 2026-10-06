; Inno Setup script: wraps dist\PlateScanner into a single PlateScanner-Setup.exe.
; Built by the GitHub Actions workflow: iscc /DAppVersion=1.0.0 installer.iss
#ifndef AppVersion
  #define AppVersion "1.0.0"
#endif

[Setup]
AppId={{7C5E1B52-4D0A-4B7E-9A55-6A2F0B7D3C11}
AppName=PSAU Gate Plate Scanner
AppVersion={#AppVersion}
AppPublisher=PSAU Security
DefaultDirName={autopf}\PlateScanner
DefaultGroupName=PSAU Gate Plate Scanner
PrivilegesRequired=lowest
DisableProgramGroupPage=yes
OutputDir=installer_output
OutputBaseFilename=PlateScanner-Setup
SetupIconFile=platescanner\assets\app.ico
UninstallDisplayIcon={app}\PlateScanner.exe
Compression=lzma2
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
WizardStyle=modern

[Tasks]
Name: "desktopicon"; Description: "Create a Desktop shortcut"; Flags: checkedonce

[Files]
Source: "dist\PlateScanner\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{autoprograms}\PSAU Gate Plate Scanner"; Filename: "{app}\PlateScanner.exe"
Name: "{autodesktop}\PSAU Gate Plate Scanner"; Filename: "{app}\PlateScanner.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\PlateScanner.exe"; Description: "Launch PSAU Gate Plate Scanner"; Flags: nowait postinstall skipifsilent
