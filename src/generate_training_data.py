from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from data_utils import FEATURE_COLUMNS, load_slope_excel, parameter_ranges_from_test_data
from morgenstern_price import SlopeInput, compute_factor_of_safety


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic slope safety training data.")
    parser.add_argument("--input", default="JIAxin WANG_create.xlsx", help="Excel test dataset path.")
    parser.add_argument("--output", default="outputs/training_data.csv", help="Output CSV path.")
    parser.add_argument("--n-samples", type=int, default=10000, help="Number of valid samples to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--num-slices", type=int, default=24, help="Slice count per slip surface.")
    parser.add_argument("--search-density", type=int, default=5, help="Slip surface search density.")
    parser.add_argument("--max-attempt-factor", type=int, default=50, help="Attempt limit multiplier.")
    return parser.parse_args()


def sample_parameters(
    rng: np.random.Generator,
    ranges: dict[str, tuple[float, float]],
) -> dict[str, float]:
    return {name: float(rng.uniform(low, high)) for name, (low, high) in ranges.items()}


def generate_training_data(args: argparse.Namespace) -> pd.DataFrame:
    test_df = load_slope_excel(args.input)
    ranges = parameter_ranges_from_test_data(test_df)
    rng = np.random.default_rng(args.seed)

    rows: list[dict[str, float | str | bool]] = []
    max_attempts = max(args.n_samples * args.max_attempt_factor, args.n_samples)
    attempts = 0

    while len(rows) < args.n_samples and attempts < max_attempts:
        attempts += 1
        params = sample_parameters(rng, ranges)
        slope = SlopeInput(
            **params,
            num_slices=args.num_slices,
            search_density=args.search_density,
        )
        result = compute_factor_of_safety(
            slope,
            candidate_limit=1,
            include_points=False,
            include_slices=False,
        )
        if not result.converged or not np.isfinite(result.fos):
            continue

        row: dict[str, float | str | bool] = {name: params[name] for name in FEATURE_COLUMNS}
        row.update(result.to_flat_dict())
        rows.append(row)

        if len(rows) % 500 == 0:
            print(f"generated {len(rows)} / {args.n_samples} valid samples")

    if len(rows) < args.n_samples:
        raise RuntimeError(
            f"Only generated {len(rows)} valid samples after {attempts} attempts. "
            "Try reducing n-samples or search-density, or widening parameter ranges."
        )

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    df = generate_training_data(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"Saved {len(df)} rows to {output}")


if __name__ == "__main__":
    main()

