# 边坡安全系数预测

这个项目用于计算和预测边坡安全系数（Factor of Safety, FOS）。它包含 Morgenstern-Price 风格的安全系数计算核心、训练数据生成脚本、随机森林预测模型训练流程，以及一个可视化的 Streamlit 本地应用。

## 下载 Windows 程序

[下载 SlopeSafetyApp.rar（约 226.5 MiB）](https://github.com/JiaxinWang-L/FACTOR-of-safe-/releases/download/v1.0.0/SlopeSafetyApp.rar) | [发布页面](https://github.com/JiaxinWang-L/FACTOR-of-safe-/releases/tag/v1.0.0) | [完整使用说明](README_WINDOWS.md)

1. 下载 `SlopeSafetyApp.rar`，使用支持 RAR 的解压工具完整解压。
2. 打开解压后的 `SlopeSafetyApp` 文件夹，双击 `SlopeSafetyApp.exe`。
3. 等待浏览器打开本地计算页面，输入边坡参数并点击“计算安全系数”。

这是 Windows 64 位免安装版，已包含 Python 和应用运行依赖。请保留与 exe 同级的 `_internal` 文件夹，并在使用期间保持启动窗口打开。GitHub 的 “Code > Download ZIP” 下载的是源码，不包含这个程序包。

本次发布上传的是已有 Windows 程序包，未重新编译；包内程序早于当前仓库源码。源码中的新增功能不保证已包含在该程序包中。批量生成数据和训练模型请参阅下方源码运行说明。

## 功能

- 计算均质边坡在圆弧滑动面下的安全系数。
- 批量生成机器学习训练数据。
- 使用随机森林回归模型预测 FOS。
- 根据 FOS 自动划分稳定性标签。
- 通过 Streamlit APP 输入边坡参数并查看滑动面可视化结果。
- 支持打包为 Windows 本地可执行程序。

## 项目结构

```text
.
├── app/
│   ├── app.py
│   ├── requirements.txt
│   └── src/
│       ├── data_utils.py
│       ├── generate_training_data.py
│       ├── morgenstern_price.py
│       └── train_random_forest.py
├── tests/
│   └── test_morgenstern_price.py
├── JIAxin WANG_create.xlsx
├── run_local_app.cmd
├── generate_10000_training_data.cmd
├── run_generate_10000.py
├── build_windows_exe.ps1
└── local_app_launcher.py
```

`build/`、`dist/`、`outputs/`、缓存文件和模型文件属于本地生成物，默认不会提交到 Git。

## 环境准备

建议使用 Conda 环境。项目脚本默认使用名为 `pytorch_wjx` 的环境：

```powershell
conda create -n pytorch_wjx python=3.11
conda activate pytorch_wjx
pip install -r app/requirements.txt
```

如果你已经有 `pytorch_wjx` 环境，可以直接安装依赖：

```powershell
conda run -n pytorch_wjx pip install -r app/requirements.txt
```

## 启动本地 APP

最简单的方式是双击根目录下的：

```text
run_local_app.cmd
```

也可以手动运行：

```powershell
cd app
conda run -n pytorch_wjx streamlit run app.py
```

启动后在浏览器打开：

```text
http://localhost:8501
```

## 生成训练数据

使用示例 Excel 数据生成 10000 条训练样本：

```powershell
generate_10000_training_data.cmd
```

或手动运行：

```powershell
cd app
conda run -n pytorch_wjx python src/generate_training_data.py --input "..\JIAxin WANG_create.xlsx" --output "..\outputs\training_data_10000.csv" --n-samples 10000 --num-slices 30 --search-density 8 --search-mode standard --progress-interval 50
```

主要输入特征包括：

```text
gamma, c, beta, phi, H, ru
```

输出会包含计算得到的安全系数和稳定性标签。

## 训练随机森林模型

生成训练数据后运行：

```powershell
cd app
conda run -n pytorch_wjx python src/train_random_forest.py --train "..\outputs\training_data_10000.csv" --test "..\JIAxin WANG_create.xlsx" --outputs "..\outputs"
```

训练输出包括：

```text
outputs/predictions.csv
outputs/metrics.json
outputs/figures/
outputs/model/random_forest_fos.joblib
```

## 稳定性分类

项目使用以下阈值划分稳定性：

```text
unstable: FOS < 1.0
critical: 1.0 <= FOS < 1.1
stable: FOS >= 1.1
```

## 运行测试

```powershell
conda run -n pytorch_wjx python -m unittest discover tests
```

## 打包 Windows 程序

如果需要生成可分享的本地程序：

```powershell
.\build_windows_exe.ps1
```

打包产物会生成到：

```text
dist/SlopeSafetyApp/
```

分享时需要发送整个 `dist/SlopeSafetyApp` 文件夹，而不是只发送单个 exe 文件。

## 注意

这个项目适合快速计算、训练数据生成和方案对比。计算结果不能直接替代工程审查、施工图设计或专业安全评估。
