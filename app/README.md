# 安全系数预测

这个项目包含三个部分：

1. 摩根斯坦-普莱斯风格的均质边坡安全系数计算核心。
2. 批量生成训练数据的脚本。
3. Streamlit 本地 APP，用于单个边坡安全系数计算和滑动面可视化。

## 环境

建议使用已有 Conda 环境：

```powershell
conda run -n pytorch_wjx pip install -r requirements.txt
```

启动 APP：

```powershell
conda run -n pytorch_wjx streamlit run app.py
```

## 生成训练数据

```powershell
conda run -n pytorch_wjx python src/generate_training_data.py --input "JIAxin WANG_create.xlsx" --output outputs/training_data.csv --n-samples 10000
```

训练数据字段为：

```text
gamma, c, beta, phi, H, ru, FOS, Stability
```

## 训练随机森林

```powershell
conda run -n pytorch_wjx python src/train_random_forest.py --train outputs/training_data.csv --test "JIAxin WANG_create.xlsx"
```

输出会保存到 `outputs/`：

```text
training_data.csv
predictions.csv
metrics.json
figures/
model/random_forest_fos.joblib
```

## 稳定性标签

```text
unstable: FOS < 1.0
critical: 1.0 <= FOS < 1.1
stable: FOS >= 1.1
```

## 说明

第一版采用简单均质边坡和圆弧滑动面搜索，适合生成机器学习训练样本和进行快速方案比较。它不是工程审查或施工图设计的替代品。
