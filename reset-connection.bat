@ECHO OFF
SETLOCAL

REM Check that PowerShell is available
WHERE powershell.exe >NUL 2>&1
IF ERRORLEVEL 1 (
    ECHO ERROR: PowerShell was not found on this machine.
    ECHO This tool requires Windows PowerShell ^(built into Windows 10^).
    PAUSE
    EXIT /B 1
)

REM Detect if already running as Administrator
NET SESSION >NUL 2>&1
IF %ERRORLEVEL% NEQ 0 (
    REM Not elevated -- re-launch this .bat via UAC
    powershell.exe -NoProfile -ExecutionPolicy Bypass -Command ^
        "Start-Process cmd.exe -ArgumentList '/c \"%~f0\"' -Verb RunAs"
    EXIT /B
)

REM Locate the .ps1 next to this .bat
SET "PS1=%~dp0reset-connection.ps1"

IF NOT EXIST "%PS1%" (
    ECHO ERROR: reset-connection.ps1 not found.
    ECHO Expected location: %PS1%
    PAUSE
    EXIT /B 1
)

REM Run the reset script
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%PS1%"

ENDLOCAL
EXIT /B %ERRORLEVEL%
