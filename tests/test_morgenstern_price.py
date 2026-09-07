import math
from pathlib import Path
import sys
import time
import unittest

import numpy as np

APP_DIR = Path(__file__).resolve().parents[1] / "app"
for import_path in (APP_DIR, APP_DIR / "src"):
    path_text = str(import_path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

from src.data_utils import FEATURE_COLUMNS, load_slope_excel
from src.morgenstern_price import SlopeInput, compute_factor_of_safety


def effective_platform_widths(slope: SlopeInput) -> tuple[float, float]:
    length = slope.H / math.tan(math.radians(slope.beta))
    toe_width = slope.toe_width if slope.toe_width > 0 else max(slope.H * 0.65, length * 0.30)
    crest_width = slope.crest_width if slope.crest_width > 0 else max(slope.H, length * 0.45)
    return toe_width, crest_width


def effective_foundation_depth(slope: SlopeInput) -> float:
    return slope.foundation_depth if slope.foundation_depth > 0 else max(slope.H * 0.35, 1.0)


def ground_y(slope: SlopeInput, x: float) -> float:
    length = slope.H / math.tan(math.radians(slope.beta))
    _, crest_width = effective_platform_widths(slope)
    toe_x = crest_width + length
    toe_y = effective_foundation_depth(slope)
    crest_y = toe_y + slope.H
    if x <= crest_width:
        return crest_y
    if x >= toe_x:
        return toe_y
    return crest_y - (x - crest_width) * math.tan(math.radians(slope.beta))


class MorgensternPriceCoreTest(unittest.TestCase):
    def test_compute_returns_valid_controlling_surface(self):
        slope = SlopeInput(
            gamma=18.0,
            c=20.0,
            beta=35.0,
            phi=28.0,
            H=20.0,
            ru=0.1,
            num_slices=24,
            search_density=4,
            search_mode="standard",
        )
        result = compute_factor_of_safety(slope, candidate_limit=5)

        self.assertTrue(result.converged, result.message)
        self.assertEqual(result.search_mode, "standard")
        self.assertTrue(math.isfinite(result.fos))
        self.assertGreater(result.fos, 0)
        self.assertGreater(result.searched_surfaces, result.solved_surfaces)
        self.assertGreater(result.screened_surfaces, 0)
        self.assertGreater(len(result.candidates), 0)
        self.assertIsNotNone(result.critical_surface)
        self.assertAlmostEqual(
            result.fos,
            min(surface.fos for surface in result.candidates),
            places=8,
        )

    def test_slip_surface_endpoints_follow_ground_surface(self):
        slope = SlopeInput(
            gamma=18.0,
            c=20.0,
            beta=35.0,
            phi=28.0,
            H=20.0,
            ru=0.1,
            num_slices=24,
            search_density=4,
            search_mode="fast",
            toe_width=12.0,
            crest_width=18.0,
            foundation_depth=7.0,
        )
        result = compute_factor_of_safety(slope, candidate_limit=5)
        self.assertTrue(result.converged, result.message)
        self.assertIsNotNone(result.critical_surface)
        surface = result.critical_surface

        self.assertAlmostEqual(surface.entry_y, ground_y(slope, surface.entry_x), delta=1e-5)
        self.assertAlmostEqual(surface.exit_y, ground_y(slope, surface.exit_x), delta=1e-5)
        self.assertAlmostEqual(surface.points[0][0], surface.entry_x)
        self.assertAlmostEqual(surface.points[0][1], surface.entry_y)
        self.assertAlmostEqual(surface.points[-1][0], surface.exit_x)
        self.assertAlmostEqual(surface.points[-1][1], surface.exit_y)
        for x_value, y_value in surface.points:
            self.assertGreaterEqual(y_value, -1e-5)
            self.assertLessEqual(y_value, ground_y(slope, x_value) + 1e-5)

    def test_expanded_search_can_find_lower_surface_than_legacy_platform_search(self):
        params = {
            "gamma": 19.35636771081259,
            "c": 1.957195463652837,
            "beta": 50.81912599348804,
            "phi": 31.22192484079935,
            "H": 55.23479922561183,
            "ru": 0.1991876262160132,
            "toe_width": 15.421072322787188,
            "crest_width": 7.377999738809573,
            "foundation_depth": 10.316079302733542,
            "num_slices": 18,
            "search_density": 3,
        }
        legacy = compute_factor_of_safety(
            SlopeInput(**params, search_mode="legacy"),
            candidate_limit=1,
            include_points=False,
            include_slices=False,
        )
        expanded = compute_factor_of_safety(
            SlopeInput(**params, search_mode="fast"),
            candidate_limit=1,
            include_points=False,
            include_slices=False,
        )

        self.assertTrue(legacy.converged, legacy.message)
        self.assertTrue(expanded.converged, expanded.message)
        self.assertLess(expanded.fos, legacy.fos * 0.95)
        self.assertLess(expanded.solved_surfaces, expanded.searched_surfaces)

    def test_ru_pore_forces_reduce_factor_of_safety(self):
        base = {
            "gamma": 18.0,
            "c": 12.0,
            "beta": 32.0,
            "phi": 24.0,
            "H": 25.0,
            "num_slices": 22,
            "search_density": 4,
            "search_mode": "fast",
        }
        dry = compute_factor_of_safety(SlopeInput(**base, ru=0.0), candidate_limit=3)
        wet = compute_factor_of_safety(SlopeInput(**base, ru=0.3), candidate_limit=3)

        self.assertTrue(dry.converged, dry.message)
        self.assertTrue(wet.converged, wet.message)
        self.assertLess(wet.fos, dry.fos)
        self.assertTrue(all(abs(item.pore_force) < 1e-9 for item in dry.critical_surface.slices))
        self.assertTrue(any(item.pore_force > 0 for item in wet.critical_surface.slices))

    def test_excel_reference_subset_remains_calculable(self):
        df = load_slope_excel(Path(__file__).resolve().parents[1] / "JIAxin WANG_create.xlsx").head(8)
        rows = []
        for _, record in df.iterrows():
            slope = SlopeInput(
                **{column: float(record[column]) for column in FEATURE_COLUMNS},
                num_slices=18,
                search_density=3,
                search_mode="fast",
            )
            result = compute_factor_of_safety(
                slope,
                candidate_limit=1,
                include_points=False,
                include_slices=False,
            )
            rows.append((float(record["FOS"]), result.fos if result.converged else float("nan")))

        values = np.asarray(rows, dtype=float)
        valid = np.isfinite(values[:, 1])
        diff = values[valid, 1] - values[valid, 0]
        self.assertGreaterEqual(int(valid.sum()), 6)
        self.assertLess(abs(float(diff.mean())), 1.2)
        self.assertLess(float(np.mean(np.abs(diff))), 1.8)
        self.assertLessEqual(int((diff > 0).sum()), 5)

    def test_standard_mode_performance_for_typical_case(self):
        slope = SlopeInput(
            gamma=18.0,
            c=20.0,
            beta=35.0,
            phi=28.0,
            H=20.0,
            ru=0.1,
            num_slices=24,
            search_density=4,
            search_mode="standard",
        )
        start = time.perf_counter()
        result = compute_factor_of_safety(
            slope,
            candidate_limit=5,
            include_points=False,
            include_slices=False,
        )
        elapsed = time.perf_counter() - start

        self.assertTrue(result.converged, result.message)
        self.assertLess(elapsed, 12.0)


if __name__ == "__main__":
    unittest.main()
