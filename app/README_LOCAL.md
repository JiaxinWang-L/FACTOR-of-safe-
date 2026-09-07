# Local Windows App

This folder contains the Streamlit app and the calculation modules.

## Fast local start

From the project root, double-click:

```text
run_local_app.cmd
```

It uses the existing `pytorch_wjx` conda environment and opens:

```text
http://localhost:8501
```

## Build Windows exe

From PowerShell in the project root:

```powershell
.\build_windows_exe.ps1
```

The recommended distributable output is:

```text
dist\SlopeSafetyApp\SlopeSafetyApp.exe
```

Keep the whole `dist\SlopeSafetyApp` folder together when sharing it.

