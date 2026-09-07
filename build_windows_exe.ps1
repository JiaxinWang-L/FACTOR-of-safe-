$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$AppDir = Join-Path $Root "app"
$Launcher = Join-Path $Root "local_app_launcher.py"
$Python = "D:\Users\admin\anaconda3\envs\pytorch_wjx\python.exe"

if (!(Test-Path $Python)) {
    throw "Could not find pytorch_wjx Python at $Python"
}

if (!(Test-Path (Join-Path $AppDir "app.py"))) {
    throw "Could not find app/app.py"
}

Write-Host "Checking PyInstaller..."
$PreviousErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& $Python -m pip show pyinstaller > $null 2> $null
$HasPyInstaller = $LASTEXITCODE -eq 0
$ErrorActionPreference = $PreviousErrorActionPreference
if (-not $HasPyInstaller) {
    Write-Host "Installing PyInstaller into pytorch_wjx..."
    & $Python -m pip install pyinstaller
}

Write-Host "Building folder-based executable..."
$ExcludeModules = @(
    "torch",
    "torchvision",
    "torchaudio",
    "scipy",
    "sklearn",
    "matplotlib",
    "IPython",
    "jupyter",
    "notebook",
    "PyQt5",
    "PyQt6",
    "PySide2",
    "PySide6",
    "tensorflow",
    "tensorboard",
    "sympy"
)

$PyInstallerArgs = @(
    "--noconfirm",
    "--clean",
    "--name", "SlopeSafetyApp",
    "--add-data", "$AppDir;app",
    "--collect-all", "streamlit",
    "--collect-all", "plotly",
    "--hidden-import", "streamlit.web.bootstrap",
    "--hidden-import", "tornado.platform.asyncio"
)

foreach ($ModuleName in $ExcludeModules) {
    $PyInstallerArgs += @("--exclude-module", $ModuleName)
}

$PyInstallerArgs += $Launcher

& $Python -m PyInstaller @PyInstallerArgs

if ($LASTEXITCODE -ne 0) {
    throw "Folder-based PyInstaller build failed."
}

$ExePath = Join-Path $Root "dist\SlopeSafetyApp\SlopeSafetyApp.exe"
if (!(Test-Path $ExePath)) {
    throw "Expected executable was not created: $ExePath"
}

Write-Host ""
Write-Host "Folder-based app created:"
Write-Host $ExePath
Write-Host ""
Write-Host "This is the recommended build for sharing."
Write-Host "Optional one-file build command:"
Write-Host "& `"$Python`" -m PyInstaller --noconfirm --clean --onefile --name SlopeSafetyApp --add-data `"$AppDir;app`" --collect-all streamlit --collect-all plotly --hidden-import streamlit.web.bootstrap --hidden-import tornado.platform.asyncio --exclude-module torch --exclude-module torchvision --exclude-module torchaudio --exclude-module scipy --exclude-module sklearn --exclude-module matplotlib --exclude-module IPython --exclude-module PyQt6 `"$Launcher`""
