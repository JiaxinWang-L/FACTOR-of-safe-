from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import degrees, isfinite, radians, sqrt, tan
from typing import Iterable

import numpy as np


LABEL_UNSTABLE = "unstable"
LABEL_CRITICAL = "critical"
LABEL_STABLE = "stable"
SEARCH_MODES = {"legacy", "fast", "standard", "accurate"}


@dataclass(frozen=True)
class SlopeInput:
    gamma: float
    c: float
    beta: float
    phi: float
    H: float
    ru: float = 0.0
    num_slices: int = 30
    search_density: int = 8
    toe_width: float = 0.0
    crest_width: float = 0.0
    foundation_depth: float = 0.0
    search_mode: str = "accurate"


@dataclass
class SliceData:
    x_left: float
    x_right: float
    x_mid: float
    ground_y: float
    base_y: float
    height: float
    alpha_deg: float
    weight: float
    base_length: float = 0.0
    normal_force: float = 0.0
    shear_force: float = 0.0
    pore_force: float = 0.0


@dataclass
class SlipSurface:
    fos: float
    center_x: float
    center_y: float
    radius: float
    exit_x: float
    exit_y: float
    entry_x: float
    entry_y: float
    sagitta: float
    lambda_mp: float = 0.0
    force_residual: float = 0.0
    moment_residual: float = 0.0
    method: str = "gle"
    points: list[tuple[float, float]] = field(default_factory=list)
    slices: list[SliceData] = field(default_factory=list)

    def summary_dict(self) -> dict[str, float | str]:
        return {
            "fos": self.fos,
            "center_x": self.center_x,
            "center_y": self.center_y,
            "radius": self.radius,
            "exit_x": self.exit_x,
            "exit_y": self.exit_y,
            "entry_x": self.entry_x,
            "entry_y": self.entry_y,
            "sagitta": self.sagitta,
            "lambda_mp": self.lambda_mp,
            "force_residual": self.force_residual,
            "moment_residual": self.moment_residual,
            "method": self.method,
        }


@dataclass
class SafetyResult:
    fos: float
    label: str
    converged: bool
    message: str
    critical_surface: SlipSurface | None = None
    candidates: list[SlipSurface] = field(default_factory=list)
    search_mode: str = "accurate"
    searched_surfaces: int = 0
    screened_surfaces: int = 0
    refined_surfaces: int = 0
    solved_surfaces: int = 0
    local_refined: bool = False

    def to_flat_dict(self) -> dict[str, float | str | bool | int]:
        data: dict[str, float | str | bool | int] = {
            "FOS": self.fos,
            "Stability": self.label,
            "converged": self.converged,
            "message": self.message,
            "search_mode": self.search_mode,
            "searched_surfaces": self.searched_surfaces,
            "screened_surfaces": self.screened_surfaces,
            "refined_surfaces": self.refined_surfaces,
            "solved_surfaces": self.solved_surfaces,
            "local_refined": self.local_refined,
        }
        if self.critical_surface is not None:
            for key, value in self.critical_surface.summary_dict().items():
                data[f"critical_{key}"] = value
        return data


@dataclass(frozen=True)
class _SurfaceSpec:
    center_x: float
    center_y: float
    radius: float
    exit_x: float
    exit_y: float
    entry_x: float
    entry_y: float
    sagitta: float
    arc_sign: float


@dataclass(frozen=True)
class _MPState:
    fos: float
    lambda_mp: float
    normals: np.ndarray
    shears: np.ndarray
    interslice_normals: np.ndarray
    pore_forces: np.ndarray
    force_residual: float
    moment_residual: float


@dataclass(frozen=True)
class _SliceGeometry:
    xs: np.ndarray
    mids: np.ndarray
    widths: np.ndarray
    ground_mid: np.ndarray
    base_mid: np.ndarray
    heights: np.ndarray
    alpha: np.ndarray
    weights: np.ndarray
    base_lengths: np.ndarray
    pore_forces: np.ndarray


@dataclass(frozen=True)
class _SearchConfig:
    mode: str
    endpoint_count: int
    sagitta_count: int
    screen_slices: int
    coarse_keep: int
    refine_from: int
    refine_grid: int
    gle_candidates: int
    final_keep: int


@dataclass(frozen=True)
class _ScreenedSurface:
    fos: float
    surface: _SurfaceSpec


@dataclass(frozen=True)
class _ScreenBatch:
    best: list[_ScreenedSurface]
    attempted: int
    valid: int


def stability_label(fos: float) -> str:
    if not isfinite(fos):
        return "invalid"
    if fos < 1.0:
        return LABEL_UNSTABLE
    if fos < 1.1:
        return LABEL_CRITICAL
    return LABEL_STABLE


def compute_factor_of_safety(
    slope: SlopeInput,
    *,
    candidate_limit: int = 10,
    include_points: bool = True,
    include_slices: bool = True,
) -> SafetyResult:
    """Compute the controlling circular slip surface for a homogeneous slope.

    The default solver uses a two-stage circular search. Many ground-exiting
    trial surfaces are first screened with a fast simplified Bishop solve, then
    the most critical candidates are refined locally and solved with the
    Morgenstern-Price/GLE iteration.
    """

    validation_error = _validate_slope(slope)
    if validation_error:
        return SafetyResult(
            fos=float("nan"),
            label="invalid",
            converged=False,
            message=validation_error,
            search_mode=_normalized_search_mode(slope.search_mode),
        )

    mode = _normalized_search_mode(slope.search_mode)
    if mode == "legacy":
        return _compute_legacy_factor_of_safety(
            slope,
            candidate_limit=candidate_limit,
            include_points=include_points,
            include_slices=include_slices,
        )

    config = _search_config(slope, candidate_limit)
    coarse_batch = _screen_surfaces(
        slope,
        _candidate_surfaces(slope, config),
        screen_slices=config.screen_slices,
        keep_limit=config.coarse_keep,
    )
    screened = list(coarse_batch.best)
    searched_surfaces = coarse_batch.attempted
    screened_surfaces = coarse_batch.valid
    refined_surfaces = 0
    local_refined = False

    if config.refine_from > 0 and screened:
        refined_specs = _refined_surfaces(
            slope,
            screened[: config.refine_from],
            config=config,
        )
        refined_batch = _screen_surfaces(
            slope,
            refined_specs,
            screen_slices=config.screen_slices,
            keep_limit=config.final_keep,
            seen_keys={_surface_key(item.surface) for item in screened},
        )
        searched_surfaces += refined_batch.attempted
        screened_surfaces += refined_batch.valid
        refined_surfaces = refined_batch.attempted
        local_refined = refined_batch.attempted > 0
        screened = _merge_screened(screened + refined_batch.best, config.final_keep)
    else:
        screened = _merge_screened(screened, config.final_keep)

    if not screened:
        return SafetyResult(
            fos=float("nan"),
            label="invalid",
            converged=False,
            message="No valid slip surface found during Bishop screening.",
            search_mode=config.mode,
            searched_surfaces=searched_surfaces,
            screened_surfaces=screened_surfaces,
            refined_surfaces=refined_surfaces,
            solved_surfaces=0,
            local_refined=local_refined,
        )

    candidates: list[SlipSurface] = []
    for item in screened[: config.gle_candidates]:
        solved = _solve_surface(
            slope,
            item.surface,
            include_points=include_points,
            include_slices=include_slices,
            method="gle",
        )
        if solved is not None and isfinite(solved.fos) and solved.fos > 0:
            candidates.append(solved)

    message = "OK"
    if not candidates:
        message = "Used Bishop fallback because GLE did not converge for screened surfaces."
        for item in screened[: max(candidate_limit, min(12, len(screened)))]:
            solved = _solve_surface(
                slope,
                item.surface,
                include_points=include_points,
                include_slices=include_slices,
                method="bishop",
            )
            if solved is not None and isfinite(solved.fos) and solved.fos > 0:
                candidates.append(solved)

    if not candidates:
        return SafetyResult(
            fos=float("nan"),
            label="invalid",
            converged=False,
            message="No valid slip surface found after GLE solve.",
            search_mode=config.mode,
            searched_surfaces=searched_surfaces,
            screened_surfaces=screened_surfaces,
            refined_surfaces=refined_surfaces,
            solved_surfaces=0,
            local_refined=local_refined,
        )

    candidates.sort(key=lambda item: item.fos)
    critical = candidates[0]
    kept = candidates[: max(candidate_limit, 1)]
    return SafetyResult(
        fos=critical.fos,
        label=stability_label(critical.fos),
        converged=True,
        message=message,
        critical_surface=critical,
        candidates=kept,
        search_mode=config.mode,
        searched_surfaces=searched_surfaces,
        screened_surfaces=screened_surfaces,
        refined_surfaces=refined_surfaces,
        solved_surfaces=len(candidates),
        local_refined=local_refined,
    )


def result_to_dict(result: SafetyResult) -> dict:
    data = result.to_flat_dict()
    data["candidates"] = [surface.summary_dict() for surface in result.candidates]
    if result.critical_surface is not None:
        data["critical_surface"] = result.critical_surface.summary_dict()
        data["critical_slices"] = [asdict(item) for item in result.critical_surface.slices]
    return data


def _compute_legacy_factor_of_safety(
    slope: SlopeInput,
    *,
    candidate_limit: int,
    include_points: bool,
    include_slices: bool,
) -> SafetyResult:
    candidates: list[SlipSurface] = []
    searched = 0
    for surface in _legacy_candidate_surfaces(slope):
        searched += 1
        solved = _solve_surface(
            slope,
            surface,
            include_points=include_points,
            include_slices=include_slices,
            method="gle",
        )
        if solved is not None and isfinite(solved.fos) and solved.fos > 0:
            candidates.append(solved)

    if not candidates:
        return SafetyResult(
            fos=float("nan"),
            label="invalid",
            converged=False,
            message="No valid slip surface found for these parameters.",
            search_mode="legacy",
            searched_surfaces=searched,
        )

    candidates.sort(key=lambda item: item.fos)
    critical = candidates[0]
    kept = candidates[: max(candidate_limit, 1)]
    return SafetyResult(
        fos=critical.fos,
        label=stability_label(critical.fos),
        converged=True,
        message="OK",
        critical_surface=critical,
        candidates=kept,
        search_mode="legacy",
        searched_surfaces=searched,
        screened_surfaces=len(candidates),
        solved_surfaces=len(candidates),
    )


def _validate_slope(slope: SlopeInput) -> str | None:
    checks = {
        "gamma": slope.gamma,
        "c": slope.c,
        "beta": slope.beta,
        "phi": slope.phi,
        "H": slope.H,
        "ru": slope.ru,
        "toe_width": slope.toe_width,
        "crest_width": slope.crest_width,
        "foundation_depth": slope.foundation_depth,
    }
    for name, value in checks.items():
        if not isfinite(float(value)):
            return f"{name} must be finite."
    if slope.gamma <= 0:
        return "gamma must be positive."
    if slope.c < 0:
        return "c must be non-negative."
    if not 3.0 <= slope.beta <= 80.0:
        return "beta must be between 3 and 80 degrees."
    if not 0.0 <= slope.phi <= 70.0:
        return "phi must be between 0 and 70 degrees."
    if slope.H <= 0:
        return "H must be positive."
    if not 0.0 <= slope.ru <= 0.95:
        return "ru must be between 0 and 0.95."
    if slope.num_slices < 8:
        return "num_slices must be at least 8."
    if slope.search_density < 3:
        return "search_density must be at least 3."
    if slope.toe_width < 0:
        return "toe_width must be non-negative."
    if slope.crest_width < 0:
        return "crest_width must be non-negative."
    if slope.foundation_depth < 0:
        return "foundation_depth must be non-negative."
    if _normalized_search_mode(slope.search_mode) not in SEARCH_MODES:
        return f"search_mode must be one of: {', '.join(sorted(SEARCH_MODES))}."
    return None


def _normalized_search_mode(search_mode: str) -> str:
    mode = str(search_mode or "accurate").strip().lower()
    if mode in {"precise", "fine"}:
        return "accurate"
    if mode in {"normal", "default"}:
        return "standard"
    return mode


def _search_config(slope: SlopeInput, candidate_limit: int) -> _SearchConfig:
    mode = _normalized_search_mode(slope.search_mode)
    density = int(slope.search_density)
    limit = max(int(candidate_limit), 1)

    if mode == "fast":
        endpoint_count = max(density * 2 + 9, 17)
        sagitta_count = max(density * 2 + 7, 13)
        screen_slices = min(max(10, slope.num_slices // 2), 18)
        return _SearchConfig(
            mode=mode,
            endpoint_count=endpoint_count,
            sagitta_count=sagitta_count,
            screen_slices=screen_slices,
            coarse_keep=max(24, limit * 5),
            refine_from=0,
            refine_grid=0,
            gle_candidates=max(16, limit * 3),
            final_keep=max(24, limit * 5),
        )

    if mode == "standard":
        endpoint_count = max(density * 3 + 13, 25)
        sagitta_count = max(density * 3 + 9, 19)
        screen_slices = min(max(14, slope.num_slices // 2), 24)
        return _SearchConfig(
            mode=mode,
            endpoint_count=endpoint_count,
            sagitta_count=sagitta_count,
            screen_slices=screen_slices,
            coarse_keep=max(48, limit * 6),
            refine_from=max(6, limit),
            refine_grid=3,
            gle_candidates=max(24, limit * 4),
            final_keep=max(48, limit * 6),
        )

    endpoint_count = max(density * 4 + 17, 35)
    sagitta_count = max(density * 3 + 15, 27)
    screen_slices = min(max(18, int(round(slope.num_slices * 0.67))), 30)
    return _SearchConfig(
        mode="accurate",
        endpoint_count=endpoint_count,
        sagitta_count=sagitta_count,
        screen_slices=screen_slices,
        coarse_keep=max(80, limit * 8),
        refine_from=max(10, limit),
        refine_grid=5,
        gle_candidates=max(36, limit * 5),
        final_keep=max(80, limit * 8),
    )


def _slope_length(slope: SlopeInput) -> float:
    return slope.H / tan(radians(slope.beta))


def _platform_widths(slope: SlopeInput) -> tuple[float, float]:
    length = _slope_length(slope)
    toe_width = slope.toe_width if slope.toe_width > 0 else max(slope.H * 0.65, length * 0.30)
    crest_width = slope.crest_width if slope.crest_width > 0 else max(slope.H, length * 0.45)
    return toe_width, crest_width


def _foundation_depth(slope: SlopeInput) -> float:
    return slope.foundation_depth if slope.foundation_depth > 0 else max(slope.H * 0.35, 1.0)


def _domain_points(slope: SlopeInput) -> tuple[float, float, float, float]:
    length = _slope_length(slope)
    toe_width, crest_width = _platform_widths(slope)
    toe_x = crest_width + length
    right_x = toe_x + toe_width
    return 0.0, crest_width, toe_x, right_x


def _ground_y(slope: SlopeInput, x: float | np.ndarray) -> float | np.ndarray:
    x_array = np.asarray(x, dtype=float)
    _, crest_x, toe_x, _ = _domain_points(slope)
    toe_y = _foundation_depth(slope)
    crest_y = toe_y + slope.H
    y = np.where(
        x_array <= crest_x,
        crest_y,
        np.where(
            x_array >= toe_x,
            toe_y,
            crest_y - (x_array - crest_x) * tan(radians(slope.beta)),
        ),
    )
    if np.isscalar(x):
        return float(y)
    return y


def _candidate_surfaces(slope: SlopeInput, config: _SearchConfig) -> Iterable[_SurfaceSpec]:
    sample_xs = _ground_sample_xs(slope, config.endpoint_count)
    min_span = _minimum_surface_span(slope)
    sagittas = _sagitta_values(slope, config.sagitta_count)

    for entry_index, entry_x in enumerate(sample_xs[:-1]):
        for exit_x in sample_xs[entry_index + 1 :]:
            if exit_x - entry_x < min_span:
                continue
            entry_y = float(_ground_y(slope, entry_x))
            exit_y = float(_ground_y(slope, exit_x))
            if entry_y < exit_y - 1e-8:
                continue
            for sagitta in sagittas:
                surface = _surface_from_endpoints(slope, float(entry_x), float(exit_x), float(sagitta))
                if surface is not None:
                    yield surface


def _legacy_candidate_surfaces(slope: SlopeInput) -> Iterable[_SurfaceSpec]:
    length = _slope_length(slope)
    toe_width, crest_width = _platform_widths(slope)
    toe_x = crest_width + length
    density = int(slope.search_density)
    endpoint_count = max(density * 2 + 3, 9)
    sagitta_count = max(density * 2 + 3, 9)
    entry_margin = max(crest_width * 0.04, min(length, slope.H) * 0.01, 1e-3)
    exit_margin = max(toe_width * 0.04, min(length, slope.H) * 0.01, 1e-3)
    entry_start = min(entry_margin, crest_width * 0.45)
    entry_end = max(crest_width - entry_margin, entry_start + 1e-3)
    exit_start = toe_x + min(exit_margin, toe_width * 0.45)
    exit_end = max(toe_x + toe_width - exit_margin, exit_start + 1e-3)

    entry_xs = _biased_platform_samples(entry_start, entry_end, endpoint_count)
    exit_xs = _biased_platform_samples(exit_start, exit_end, endpoint_count)
    sagitta_fracs = np.linspace(0.04, 0.80, sagitta_count)

    for entry_x in entry_xs:
        for exit_x in exit_xs:
            for sagitta_frac in sagitta_fracs:
                surface = _surface_from_endpoints(
                    slope,
                    float(entry_x),
                    float(exit_x),
                    float(slope.H * sagitta_frac),
                )
                if surface is not None:
                    yield surface


def _ground_sample_xs(slope: SlopeInput, count: int) -> np.ndarray:
    left_x, crest_x, toe_x, right_x = _domain_points(slope)
    count = max(int(count), 9)
    top_count = max(4, count // 4)
    slope_count = max(7, count // 2)
    toe_count = max(4, count - top_count - slope_count + 2)

    values: list[float] = []
    values.extend(np.linspace(left_x, crest_x, top_count))
    values.extend(np.linspace(crest_x, toe_x, slope_count))
    values.extend(np.linspace(toe_x, right_x, toe_count))

    step = right_x / max(count - 1, 1)
    for anchor in (crest_x, toe_x):
        for offset in (-1.0, -0.5, 0.0, 0.5, 1.0):
            values.append(anchor + offset * step)

    clipped = np.clip(np.asarray(values, dtype=float), left_x, right_x)
    rounded = np.unique(np.round(clipped, 8))
    return np.asarray(sorted(float(value) for value in rounded), dtype=float)


def _minimum_surface_span(slope: SlopeInput) -> float:
    _, _, _, right_x = _domain_points(slope)
    return max(min(right_x * 0.025, slope.H * 0.20), 0.25)


def _sagitta_values(slope: SlopeInput, count: int) -> np.ndarray:
    count = max(int(count), 7)
    t_values = np.linspace(0.0, 1.0, count)
    shallow_biased = 0.015 + (1.15 - 0.015) * np.power(t_values, 1.55)
    return np.unique(np.maximum(shallow_biased * slope.H, slope.H * 0.003))


def _surface_from_endpoints(
    slope: SlopeInput,
    entry_x: float,
    exit_x: float,
    sagitta: float,
) -> _SurfaceSpec | None:
    if exit_x <= entry_x:
        return None

    entry_y = float(_ground_y(slope, entry_x))
    exit_y = float(_ground_y(slope, exit_x))
    p0 = np.array([entry_x, entry_y], dtype=float)
    p1 = np.array([exit_x, exit_y], dtype=float)
    chord = p1 - p0
    chord_len = float(np.linalg.norm(chord))
    if chord_len <= 1e-9:
        return None

    sagitta = min(float(sagitta), chord_len * 0.90)
    if sagitta <= 0:
        return None

    unit = chord / chord_len
    below_normal = np.array([unit[1], -unit[0]])
    midpoint = (p0 + p1) / 2.0
    radius = chord_len**2 / (8.0 * sagitta) + sagitta / 2.0
    center = midpoint - below_normal * (radius - sagitta)
    arc_sign = _matching_arc_sign(
        center_x=float(center[0]),
        center_y=float(center[1]),
        radius=float(radius),
        entry_x=entry_x,
        entry_y=entry_y,
        exit_x=exit_x,
        exit_y=exit_y,
    )
    if arc_sign is None:
        return None

    return _SurfaceSpec(
        center_x=float(center[0]),
        center_y=float(center[1]),
        radius=float(radius),
        exit_x=float(exit_x),
        exit_y=exit_y,
        entry_x=float(entry_x),
        entry_y=entry_y,
        sagitta=sagitta,
        arc_sign=arc_sign,
    )


def _biased_platform_samples(start: float, end: float, count: int) -> np.ndarray:
    if count <= 2 or end <= start:
        return np.array([(start + end) / 2.0])
    linear = np.linspace(start, end, count)
    t = np.linspace(0.0, 1.0, count)
    cosine = start + (end - start) * (1.0 - np.cos(np.pi * t)) / 2.0
    return np.unique(np.concatenate([linear, cosine]))


def _matching_arc_sign(
    *,
    center_x: float,
    center_y: float,
    radius: float,
    entry_x: float,
    entry_y: float,
    exit_x: float,
    exit_y: float,
) -> float | None:
    signs = (-1.0, 1.0)
    tolerance = max(radius * 1e-6, 1e-5)
    for sign in signs:
        entry_calc = _circle_y_from_values(center_x, center_y, radius, entry_x, sign)
        exit_calc = _circle_y_from_values(center_x, center_y, radius, exit_x, sign)
        if (
            isfinite(entry_calc)
            and isfinite(exit_calc)
            and abs(entry_calc - entry_y) <= tolerance
            and abs(exit_calc - exit_y) <= tolerance
        ):
            return sign
    return None


def _refined_surfaces(
    slope: SlopeInput,
    seeds: list[_ScreenedSurface],
    *,
    config: _SearchConfig,
) -> Iterable[_SurfaceSpec]:
    _, _, _, right_x = _domain_points(slope)
    min_span = _minimum_surface_span(slope)
    endpoint_span = right_x / max(config.endpoint_count - 1, 1) * 1.25
    sagitta_span = slope.H * 1.15 / max(config.sagitta_count - 1, 1) * 1.25
    offsets = np.linspace(-1.0, 1.0, config.refine_grid)

    yielded: set[tuple[float, float, float]] = set()
    for seed in seeds:
        base = seed.surface
        for entry_offset in offsets:
            entry_x = float(np.clip(base.entry_x + entry_offset * endpoint_span, 0.0, right_x))
            for exit_offset in offsets:
                exit_x = float(np.clip(base.exit_x + exit_offset * endpoint_span, 0.0, right_x))
                if exit_x - entry_x < min_span:
                    continue
                for sagitta_offset in offsets:
                    sagitta = max(base.sagitta + sagitta_offset * sagitta_span, slope.H * 0.003)
                    key = (round(entry_x, 6), round(exit_x, 6), round(sagitta, 6))
                    if key in yielded:
                        continue
                    yielded.add(key)
                    surface = _surface_from_endpoints(slope, entry_x, exit_x, sagitta)
                    if surface is not None:
                        yield surface


def _screen_surfaces(
    slope: SlopeInput,
    surfaces: Iterable[_SurfaceSpec],
    *,
    screen_slices: int,
    keep_limit: int,
    seen_keys: set[tuple[float, float, float]] | None = None,
) -> _ScreenBatch:
    best: list[_ScreenedSurface] = []
    seen = set(seen_keys or set())
    attempted = 0
    valid = 0

    for surface in surfaces:
        key = _surface_key(surface)
        if key in seen:
            continue
        seen.add(key)
        attempted += 1
        fos = _screen_surface(slope, surface, slice_count=screen_slices)
        if fos is None or not isfinite(fos) or fos <= 0:
            continue
        valid += 1
        best.append(_ScreenedSurface(fos=fos, surface=surface))
        if len(best) > keep_limit * 2:
            best = _merge_screened(best, keep_limit)

    return _ScreenBatch(best=_merge_screened(best, keep_limit), attempted=attempted, valid=valid)


def _merge_screened(
    surfaces: list[_ScreenedSurface],
    keep_limit: int,
) -> list[_ScreenedSurface]:
    by_key: dict[tuple[float, float, float], _ScreenedSurface] = {}
    for item in surfaces:
        key = _surface_key(item.surface)
        existing = by_key.get(key)
        if existing is None or item.fos < existing.fos:
            by_key[key] = item
    return sorted(by_key.values(), key=lambda item: item.fos)[: max(keep_limit, 1)]


def _surface_key(surface: _SurfaceSpec) -> tuple[float, float, float]:
    return (round(surface.entry_x, 5), round(surface.exit_x, 5), round(surface.sagitta, 5))


def _screen_surface(
    slope: SlopeInput,
    surface: _SurfaceSpec,
    *,
    slice_count: int,
) -> float | None:
    geometry = _slice_geometry(slope, surface, slice_count=slice_count)
    if geometry is None:
        return None

    bishop = _bishop_seed(
        weights=geometry.weights,
        base_lengths=geometry.base_lengths,
        pore_forces=geometry.pore_forces,
        alpha=geometry.alpha,
        c=slope.c,
        phi_deg=slope.phi,
        max_iter=35,
        tol=2e-4,
    )
    if isfinite(bishop) and bishop > 0:
        return bishop

    ordinary = _ordinary_fellenius_fos(
        weights=geometry.weights,
        base_lengths=geometry.base_lengths,
        pore_forces=geometry.pore_forces,
        alpha=geometry.alpha,
        c=slope.c,
        phi_deg=slope.phi,
    )
    if isfinite(ordinary) and ordinary > 0:
        return ordinary
    return None


def _solve_surface(
    slope: SlopeInput,
    surface: _SurfaceSpec,
    *,
    include_points: bool,
    include_slices: bool,
    method: str,
) -> SlipSurface | None:
    geometry = _slice_geometry(slope, surface, slice_count=slope.num_slices)
    if geometry is None:
        return None

    if method == "bishop":
        state = _bishop_state(
            geometry=geometry,
            c=slope.c,
            phi_deg=slope.phi,
        )
    else:
        state = _morgenstern_price_gle(
            weights=geometry.weights,
            base_lengths=geometry.base_lengths,
            pore_forces=geometry.pore_forces,
            alpha=geometry.alpha,
            c=slope.c,
            phi_deg=slope.phi,
        )
    if state is None or not isfinite(state.fos) or state.fos <= 0:
        return None

    points: list[tuple[float, float]] = []
    slices: list[SliceData] = []
    if include_points:
        px = np.linspace(surface.entry_x, surface.exit_x, 180)
        py = _circle_y(surface, px)
        ground_points = _ground_y(slope, px)
        tolerance = max(slope.H, 1.0) * 1e-6
        if (
            np.any(~np.isfinite(py))
            or np.any(py < -tolerance)
            or np.any(py > ground_points + tolerance)
        ):
            return None
        points = [(float(x), float(y)) for x, y in zip(px, py) if isfinite(float(y))]

    if include_slices:
        slices = [
            SliceData(
                x_left=float(geometry.xs[i]),
                x_right=float(geometry.xs[i + 1]),
                x_mid=float(geometry.mids[i]),
                ground_y=float(geometry.ground_mid[i]),
                base_y=float(geometry.base_mid[i]),
                height=float(geometry.heights[i]),
                alpha_deg=float(degrees(geometry.alpha[i])),
                weight=float(geometry.weights[i]),
                base_length=float(geometry.base_lengths[i]),
                normal_force=float(state.normals[i]),
                shear_force=float(state.shears[i]),
                pore_force=float(geometry.pore_forces[i]),
            )
            for i in range(len(geometry.mids))
        ]

    return SlipSurface(
        fos=float(state.fos),
        center_x=surface.center_x,
        center_y=surface.center_y,
        radius=surface.radius,
        exit_x=surface.exit_x,
        exit_y=surface.exit_y,
        entry_x=surface.entry_x,
        entry_y=surface.entry_y,
        sagitta=surface.sagitta,
        lambda_mp=float(state.lambda_mp),
        force_residual=float(state.force_residual),
        moment_residual=float(state.moment_residual),
        method=method,
        points=points,
        slices=slices,
    )


def _slice_geometry(
    slope: SlopeInput,
    surface: _SurfaceSpec,
    *,
    slice_count: int,
) -> _SliceGeometry | None:
    xs = np.linspace(surface.entry_x, surface.exit_x, int(slice_count) + 1)
    mids = (xs[:-1] + xs[1:]) / 2.0
    widths = xs[1:] - xs[:-1]

    base_left = _circle_y(surface, xs[:-1])
    base_mid = _circle_y(surface, mids)
    base_right = _circle_y(surface, xs[1:])
    ground_left = _ground_y(slope, xs[:-1])
    ground_mid = _ground_y(slope, mids)
    ground_right = _ground_y(slope, xs[1:])
    height_left = ground_left - base_left
    height_mid = ground_mid - base_mid
    height_right = ground_right - base_right

    tolerance = max(slope.H, 1.0) * 1e-7
    if (
        np.any(~np.isfinite(base_mid))
        or np.any(~np.isfinite(height_mid))
        or np.any(height_left < -tolerance)
        or np.any(height_mid <= tolerance)
        or np.any(height_right < -tolerance)
        or np.any(base_left < -tolerance)
        or np.any(base_mid < -tolerance)
        or np.any(base_right < -tolerance)
    ):
        return None

    max_depth = float(np.max([np.max(height_left), np.max(height_mid), np.max(height_right)]))
    if max_depth > max(slope.H * 3.0, 5.0):
        return None

    alpha = np.arctan2(base_left - base_right, widths)
    if np.any(np.abs(alpha) >= radians(88.0)):
        return None

    base_lengths = np.hypot(widths, base_right - base_left)
    areas = widths * np.maximum(height_left + 4.0 * height_mid + height_right, 0.0) / 6.0
    weights = slope.gamma * areas
    if np.any(weights <= tolerance):
        return None

    heights = areas / np.maximum(widths, 1e-9)
    pore_forces = _ru_pore_forces(
        weights=weights,
        base_lengths=base_lengths,
        widths=widths,
        ru=slope.ru,
    )
    return _SliceGeometry(
        xs=xs,
        mids=mids,
        widths=widths,
        ground_mid=ground_mid,
        base_mid=base_mid,
        heights=heights,
        alpha=alpha,
        weights=weights,
        base_lengths=base_lengths,
        pore_forces=pore_forces,
    )


def _circle_y(surface: _SurfaceSpec, x: float | np.ndarray) -> np.ndarray:
    x_array = np.asarray(x, dtype=float)
    radicand = surface.radius**2 - (x_array - surface.center_x) ** 2
    y = np.full_like(x_array, np.nan, dtype=float)
    valid = radicand >= 0.0
    y[valid] = surface.center_y + surface.arc_sign * np.sqrt(radicand[valid])
    return y


def _circle_y_from_values(
    center_x: float,
    center_y: float,
    radius: float,
    x: float,
    arc_sign: float,
) -> float:
    radicand = radius**2 - (x - center_x) ** 2
    if radicand < 0.0:
        return float("nan")
    return center_y + arc_sign * sqrt(radicand)


def _ru_pore_forces(
    *,
    weights: np.ndarray,
    base_lengths: np.ndarray,
    widths: np.ndarray,
    ru: float,
) -> np.ndarray:
    if ru <= 0:
        return np.zeros_like(weights)

    width_ratio = base_lengths / np.maximum(widths, 1e-9)
    return np.clip(ru, 0.0, 0.95) * weights * width_ratio


def _morgenstern_price_gle(
    *,
    weights: np.ndarray,
    base_lengths: np.ndarray,
    pore_forces: np.ndarray,
    alpha: np.ndarray,
    c: float,
    phi_deg: float,
    max_iter: int = 60,
    tol: float = 1e-5,
) -> _MPState | None:
    initial_fos = _bishop_seed(
        weights=weights,
        base_lengths=base_lengths,
        pore_forces=pore_forces,
        alpha=alpha,
        c=c,
        phi_deg=phi_deg,
    )
    if not isfinite(initial_fos) or initial_fos <= 0:
        initial_fos = 1.2

    starts: list[tuple[float, float]] = []
    for fos_scale in (1.0, 0.8, 1.2, 0.6, 1.5):
        for lambda_seed in (0.0, -0.5, 0.5, -1.0, 1.0):
            starts.append((max(initial_fos * fos_scale, 0.05), lambda_seed))

    best_state: _MPState | None = None
    best_norm = float("inf")
    for fos_start, lambda_start in starts:
        solved = _solve_mp_newton(
            fos_start=fos_start,
            lambda_start=lambda_start,
            weights=weights,
            base_lengths=base_lengths,
            pore_forces=pore_forces,
            alpha=alpha,
            c=c,
            phi_deg=phi_deg,
            max_iter=max_iter,
            tol=tol,
        )
        if solved is None:
            continue
        norm = abs(solved.force_residual) + abs(solved.moment_residual)
        if norm < best_norm:
            best_state = solved
            best_norm = norm
        if norm <= tol * 10:
            return solved

    return best_state if best_norm <= 5e-4 else None


def _solve_mp_newton(
    *,
    fos_start: float,
    lambda_start: float,
    weights: np.ndarray,
    base_lengths: np.ndarray,
    pore_forces: np.ndarray,
    alpha: np.ndarray,
    c: float,
    phi_deg: float,
    max_iter: int,
    tol: float,
) -> _MPState | None:
    fos = float(np.clip(fos_start, 0.05, 10.0))
    lambda_mp = float(np.clip(lambda_start, -5.0, 5.0))
    state = _evaluate_mp_state(
        fos=fos,
        lambda_mp=lambda_mp,
        weights=weights,
        base_lengths=base_lengths,
        pore_forces=pore_forces,
        alpha=alpha,
        c=c,
        phi_deg=phi_deg,
    )
    if state is None:
        return None

    for _ in range(max_iter):
        residual = np.array([state.force_residual, state.moment_residual], dtype=float)
        norm = float(np.linalg.norm(residual))
        if norm <= tol:
            return state

        h_fos = max(abs(fos) * 1e-4, 1e-5)
        h_lambda = max(abs(lambda_mp) * 1e-4, 1e-5)
        state_f = _evaluate_mp_state(
            fos=fos + h_fos,
            lambda_mp=lambda_mp,
            weights=weights,
            base_lengths=base_lengths,
            pore_forces=pore_forces,
            alpha=alpha,
            c=c,
            phi_deg=phi_deg,
        )
        state_l = _evaluate_mp_state(
            fos=fos,
            lambda_mp=lambda_mp + h_lambda,
            weights=weights,
            base_lengths=base_lengths,
            pore_forces=pore_forces,
            alpha=alpha,
            c=c,
            phi_deg=phi_deg,
        )
        if state_f is None or state_l is None:
            return state if norm < 5e-4 else None

        jacobian = np.column_stack(
            [
                (np.array([state_f.force_residual, state_f.moment_residual]) - residual) / h_fos,
                (np.array([state_l.force_residual, state_l.moment_residual]) - residual) / h_lambda,
            ]
        )
        try:
            step = np.linalg.solve(jacobian, -residual)
        except np.linalg.LinAlgError:
            step, *_ = np.linalg.lstsq(jacobian, -residual, rcond=None)

        accepted = False
        for damping in (1.0, 0.5, 0.25, 0.125, 0.0625):
            trial_fos = float(np.clip(fos + damping * step[0], 0.05, 10.0))
            trial_lambda = float(np.clip(lambda_mp + damping * step[1], -5.0, 5.0))
            trial = _evaluate_mp_state(
                fos=trial_fos,
                lambda_mp=trial_lambda,
                weights=weights,
                base_lengths=base_lengths,
                pore_forces=pore_forces,
                alpha=alpha,
                c=c,
                phi_deg=phi_deg,
            )
            if trial is None:
                continue
            trial_norm = float(np.linalg.norm([trial.force_residual, trial.moment_residual]))
            if trial_norm < norm:
                fos = trial_fos
                lambda_mp = trial_lambda
                state = trial
                accepted = True
                break
        if not accepted:
            return state if norm < 5e-4 else None

    final_norm = abs(state.force_residual) + abs(state.moment_residual)
    return state if final_norm < 5e-4 else None


def _evaluate_mp_state(
    *,
    fos: float,
    lambda_mp: float,
    weights: np.ndarray,
    base_lengths: np.ndarray,
    pore_forces: np.ndarray,
    alpha: np.ndarray,
    c: float,
    phi_deg: float,
) -> _MPState | None:
    if not isfinite(fos) or fos <= 0:
        return None

    phi_tan = tan(radians(phi_deg))
    driving = float(np.sum(weights * np.sin(alpha)))
    if driving <= 1e-9:
        return None

    n_slices = len(weights)
    f_values = _half_sine_interslice_function(n_slices)
    normals = np.zeros(n_slices, dtype=float)
    shears = np.zeros(n_slices, dtype=float)
    interslice_normals = np.zeros(n_slices + 1, dtype=float)

    for i in range(n_slices):
        sin_alpha = float(np.sin(alpha[i]))
        cos_alpha = float(np.cos(alpha[i]))
        e_left = interslice_normals[i]
        f_left = f_values[i]
        f_right = f_values[i + 1]
        strength_constant = c * base_lengths[i] - pore_forces[i] * phi_tan
        m_alpha = cos_alpha + sin_alpha * phi_tan / fos
        q_alpha = sin_alpha - cos_alpha * phi_tan / fos
        if abs(m_alpha) <= 1e-10:
            return None

        rhs = (
            e_left
            - strength_constant * cos_alpha / fos
            + (q_alpha / m_alpha)
            * (weights[i] - lambda_mp * f_left * e_left - strength_constant * sin_alpha / fos)
        )
        denominator = 1.0 - (q_alpha / m_alpha) * lambda_mp * f_right
        if abs(denominator) <= 1e-10:
            return None

        e_right = rhs / denominator
        interslice_normals[i + 1] = e_right
        delta_x = lambda_mp * (f_right * e_right - f_left * e_left)
        normal = (weights[i] + delta_x - strength_constant * sin_alpha / fos) / m_alpha
        shear = (strength_constant + normal * phi_tan) / fos
        if not isfinite(normal) or not isfinite(shear) or normal <= 0:
            return None
        normals[i] = normal
        shears[i] = shear

    available_strength = c * base_lengths + (normals - pore_forces) * phi_tan
    if np.any(~np.isfinite(available_strength)) or np.any(available_strength <= 0):
        return None

    moment_fos = float(np.sum(available_strength) / driving)
    normalizer = max(float(np.sum(weights)), 1.0)
    force_residual = float(interslice_normals[-1] / normalizer)
    moment_residual = float((moment_fos - fos) / max(fos, 1e-9))
    if not isfinite(force_residual) or not isfinite(moment_residual):
        return None

    return _MPState(
        fos=float(fos),
        lambda_mp=float(lambda_mp),
        normals=normals,
        shears=shears,
        interslice_normals=interslice_normals,
        pore_forces=pore_forces,
        force_residual=force_residual,
        moment_residual=moment_residual,
    )


def _half_sine_interslice_function(n_slices: int) -> np.ndarray:
    positions = np.linspace(0.0, 1.0, n_slices + 1)
    values = np.sin(np.pi * positions)
    values[0] = 0.0
    values[-1] = 0.0
    return values


def _bishop_seed(
    *,
    weights: np.ndarray,
    base_lengths: np.ndarray,
    pore_forces: np.ndarray,
    alpha: np.ndarray,
    c: float,
    phi_deg: float,
    max_iter: int = 100,
    tol: float = 1e-5,
) -> float:
    phi_tan = tan(radians(phi_deg))
    driving = float(np.sum(weights * np.sin(alpha)))
    if driving <= 1e-9:
        return float("nan")

    strength_constant = c * base_lengths - pore_forces * phi_tan
    fos = 1.2
    for _ in range(max_iter):
        denominator = np.cos(alpha) + (np.sin(alpha) * phi_tan / max(fos, 1e-6))
        if np.any(np.abs(denominator) <= 1e-9):
            return float("nan")
        updated = float(np.sum((strength_constant + weights * phi_tan) / denominator) / driving)
        if not isfinite(updated) or updated <= 0:
            return float("nan")
        if abs(updated - fos) < tol:
            return updated
        fos = 0.5 * fos + 0.5 * updated
    return fos


def _bishop_state(
    *,
    geometry: _SliceGeometry,
    c: float,
    phi_deg: float,
) -> _MPState | None:
    fos = _bishop_seed(
        weights=geometry.weights,
        base_lengths=geometry.base_lengths,
        pore_forces=geometry.pore_forces,
        alpha=geometry.alpha,
        c=c,
        phi_deg=phi_deg,
    )
    if not isfinite(fos) or fos <= 0:
        return None

    phi_tan = tan(radians(phi_deg))
    normals = np.maximum(geometry.weights * np.cos(geometry.alpha), 1e-9)
    shears = (c * geometry.base_lengths + (normals - geometry.pore_forces) * phi_tan) / fos
    if np.any(~np.isfinite(shears)):
        return None

    return _MPState(
        fos=float(fos),
        lambda_mp=0.0,
        normals=normals,
        shears=shears,
        interslice_normals=np.zeros(len(geometry.weights) + 1),
        pore_forces=geometry.pore_forces,
        force_residual=float("nan"),
        moment_residual=0.0,
    )


def _ordinary_fellenius_fos(
    *,
    weights: np.ndarray,
    base_lengths: np.ndarray,
    pore_forces: np.ndarray,
    alpha: np.ndarray,
    c: float,
    phi_deg: float,
) -> float:
    phi_tan = tan(radians(phi_deg))
    driving = float(np.sum(weights * np.sin(alpha)))
    if driving <= 1e-9:
        return float("nan")
    effective_normal = weights * np.cos(alpha) - pore_forces
    available = c * base_lengths + effective_normal * phi_tan
    if np.any(~np.isfinite(available)) or float(np.sum(available)) <= 0:
        return float("nan")
    return float(np.sum(available) / driving)
