#define MyAppVersion GetEnv('APP_VERSION')
#if MyAppVersion == ""
#define MyAppVersion "0.0.0"
#endif

[Setup]
AppId={{B8F3E2A1-7C4D-4E5F-9A1B-2D3C4E5F6A7B}
AppName=CATTS - Video Transcriber
AppVersion={#MyAppVersion}
AppPublisher=CATTS
DefaultDirName={autopf}\CATTS
DefaultGroupName=CATTS - Video Transcriber
OutputDir=output\installer
OutputBaseFilename=CATTS-v{#MyAppVersion}-Setup-Windows-x64
SetupIconFile=..\assets\icon\app.ico
UninstallDisplayIcon={app}\VideoTranscriber.exe
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest

; 업데이트 지원
AppVerName=CATTS v{#MyAppVersion}
VersionInfoVersion={#MyAppVersion}.0
UsePreviousAppDir=yes
CloseApplications=yes
RestartApplications=yes

[Languages]
Name: "korean"; MessagesFile: "compiler:Languages\Korean.isl"

[Tasks]
Name: "desktopicon"; Description: "바탕화면에 CATTS 바로가기 만들기"; GroupDescription: "추가 옵션:"; Flags: unchecked

[Files]
Source: "output\dist\VideoTranscriber\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\CATTS"; Filename: "{app}\VideoTranscriber.exe"; IconFilename: "{app}\assets\icon\app.ico"
Name: "{group}\CATTS 제거"; Filename: "{uninstallexe}"
Name: "{autodesktop}\CATTS"; Filename: "{app}\VideoTranscriber.exe"; IconFilename: "{app}\assets\icon\app.ico"; Tasks: desktopicon

[Run]
Filename: "{app}\VideoTranscriber.exe"; Description: "CATTS 실행"; Flags: nowait postinstall skipifsilent
