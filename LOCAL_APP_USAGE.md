# 本地 APP 使用说明

## 方式一：本机快速运行

双击根目录里的：

```text
run_local_app.cmd
```

它会使用本机已有的 `pytorch_wjx` 环境启动 APP，并打开：

```text
http://localhost:8501
```

这个方式适合你自己的电脑。

## 方式二：exe 本地程序

已经生成的 exe 在：

```text
dist\SlopeSafetyApp\SlopeSafetyApp.exe
```

双击 `SlopeSafetyApp.exe` 后，会自动启动本地服务并打开浏览器。

分享给别人时，不要只发单独的 exe 文件，要把整个文件夹一起发：

```text
dist\SlopeSafetyApp
```

## 重新打包

如果修改了 APP 代码，重新运行：

```powershell
.\build_windows_exe.ps1
```

打包完成后，再使用新的：

```text
dist\SlopeSafetyApp\SlopeSafetyApp.exe
```

