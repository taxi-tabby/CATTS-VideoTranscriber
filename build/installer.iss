[Setup]
AppName=CATTS - Video Transcriber
AppVersion=1.0.0
AppPublisher=CATTS
DefaultDirName={autopf}\CATTS
DefaultGroupName=CATTS - Video Transcriber
OutputDir=output\installer
OutputBaseFilename=CATTS_Setup_1.0.0
SetupIconFile=..\assets\icon\app.ico
UninstallDisplayIcon={app}\VideoTranscriber.exe
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest

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
