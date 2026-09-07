from pathlib import Path
from types import SimpleNamespace
import sys


ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "app" / "src"
sys.path.insert(0, str(SRC_DIR))

from generate_training_data import generate_training_data  # noqa: E402


def main() -> None:
    output = ROOT / "outputs" / "training_data_10000.csv"
    args = SimpleNamespace(
        input=str(ROOT / "JIAxin WANG_create.xlsx"),
        output=str(output),
        n_samples=10000,
        seed=42,
        num_slices=30,
        search_density=8,
        search_mode="standard",
        max_attempt_factor=50,
        progress_interval=50,
    )

    print("程序已经开始运行。", flush=True)
    print("正在生成 10000 条训练数据，看到下面的数字增加就表示正常。", flush=True)
    print("这个过程可能需要一段时间，请不要关闭 VSCode。", flush=True)
    df = generate_training_data(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"完成。已保存 {len(df)} 条数据到：{output}", flush=True)


if __name__ == "__main__":
    main()
