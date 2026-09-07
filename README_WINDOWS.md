# SlopeSafetyApp Windows 版使用说明

## 下载

- [下载 Windows 程序包 SlopeSafetyApp.rar](https://github.com/JiaxinWang-L/FACTOR-of-safe-/releases/download/v1.0.0/SlopeSafetyApp.rar)
- [查看 v1.0.0 发布页面](https://github.com/JiaxinWang-L/FACTOR-of-safe-/releases/tag/v1.0.0)
- [查看项目与源码说明](https://github.com/JiaxinWang-L/FACTOR-of-safe-#readme)

程序包大小为 237,490,031 字节（约 226.5 MiB）。这是 Windows 64 位免安装版，包含 Python 运行环境和应用依赖，无需另外安装 Python 或 Conda。

本次上传的是已有程序包，未重新编译。包内程序早于当前仓库源码，界面和计算实现可能与源码版本不同。本说明面向该压缩包；批量数据生成及随机森林训练请按仓库主 README 的源码流程运行。

## 解压与启动

1. 下载上述 RAR 文件，使用支持 RAR 的解压工具将全部内容解压到本地文件夹。
2. 打开解压后的 `SlopeSafetyApp` 文件夹。
3. 双击 `SlopeSafetyApp.exe`，等待本地服务启动并打开浏览器。
4. 使用期间保持启动窗口打开。结束使用时，在启动窗口按 `Ctrl+C`。

解压后的主要结构：

```text
SlopeSafetyApp/
|-- SlopeSafetyApp.exe
`-- _internal/
    |-- app/
    |-- python311.dll
    `-- ...其他运行依赖
```

请保留整个文件夹。不要直接在压缩软件中运行 exe，也不要只移动 exe 或删除 `_internal`。

界面在浏览器中显示，但计算程序运行在本机。启动器从 8501 至 8599 中选择可用端口，实际地址会显示在启动窗口中。

## 计算与导出

1. 在左侧输入单位重、黏聚力、坡角、内摩擦角、坡高和孔压系数。
2. 按实际边坡设置坡顶与坡底平台宽度、底边界高度，并调整条分数量与滑面搜索精度。
3. 点击“计算安全系数”，查看 FOS、稳定性标签、候选滑动面和临界滑动面。
4. 点击“下载当前计算结果 CSV”保存计算结果。

本项目的标签规则为：

| 标签 | FOS 范围 |
| --- | --- |
| unstable | FOS < 1.0 |
| critical | 1.0 <= FOS < 1.1 |
| stable | FOS >= 1.1 |

这些标签用于本程序的结果分类，不代表具体工程的验收标准。计算结果不能直接替代工程审查或施工图设计。

## 常见问题

### 下载后只有源码，没有 exe

“Code > Download ZIP”和 Release 自动生成的 “Source code” 都是源码压缩包。请在发布页面的 Assets 中下载 `SlopeSafetyApp.rar`。

### 浏览器没有自动打开

查看启动窗口中显示的本地地址，将它输入浏览器。先确认启动窗口仍在运行，并等待首次启动完成。

### 提示缺少 DLL 或找不到应用文件

确认已经完整解压，且 exe 旁边保留了 `_internal` 文件夹。不要从压缩软件预览窗口中直接启动。

### 页面关闭后如何继续使用

启动窗口仍在运行时，重新访问其中显示的本地地址；退出程序后则重新双击 exe。只关闭浏览器标签页不会主动结束本地服务。

## 文件校验

在下载目录中用 PowerShell 运行：

```powershell
Get-FileHash .\SlopeSafetyApp.rar -Algorithm SHA256
```

本次发布原包的 SHA256：

```text
7256e290f9742b7257110bb49f98f624d51120c0efd65460da3cbbe423c855e3
```
