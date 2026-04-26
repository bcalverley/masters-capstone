@echo off
setlocal

set "EXE=%~dp0dist\PokemonCardScanner\PokemonCardScanner.exe"
set "ICON=%~dp0dist\PokemonCardScanner\PokemonCardScanner.exe"
set "WORKDIR=%~dp0dist\PokemonCardScanner"

if not exist "%EXE%" (
    echo ERROR: Executable not found. Run build.bat first.
    pause & exit /b 1
)

powershell -NoProfile -Command ^
  "$desktop = [System.Environment]::GetFolderPath('Desktop');" ^
  "$link = '$desktop\Pokemon Card Scanner.lnk';" ^
  "$wsh = New-Object -ComObject WScript.Shell;" ^
  "$sc = $wsh.CreateShortcut($link);" ^
  "$sc.TargetPath = '%EXE%';" ^
  "$sc.IconLocation = '%ICON%';" ^
  "$sc.WorkingDirectory = '%WORKDIR%';" ^
  "$sc.Save();" ^
  "Write-Host 'Shortcut created at' $link"

echo.
pause
