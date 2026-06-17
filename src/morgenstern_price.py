from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import atan, cos, degrees, isfinite, radians, sin, sqrt, tan
from typing import Iterable

import numpy as np


LABEL_UNSTABLE = "unstable"
LABEL_CRITICAL = "critical"
LABEL_STABLE = "stable"


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


@dataclass
class SlipSurface:
    fos: float
    center_x: float
    center_y: float
    radius: float
    entry_x: float
    entry_y: float
    sagitta: float
    points: list[tuple[float, float]] = field(default_factory=list)
    slices: list[SliceData] = field(default_factory=list)

    def summary_dict(self) -> dict[str, float]:
        return {
            "fos": self.fos,
            "center_x": self.center_x,
            "center_y": self.center_y,
            "radius": self.radius,
            "entry_x": self.entry_x,
            "entry_y": self.entry_y,
            "sagitta": self.sagitta,
        }


@dataclass
class SafetyResult:
    fos: float
    label: str
    converged: bool
    message: str
    critical_surface: SlipSurface | None = None
    candidates: list[SlipSurface] = field(default_factory=list)

    def to_flat_dict(self) -> dict[str, float | str | bool]:
        data: dict[str, float | str | bool] = {
            "FOS": self.fos,
            "Stability": self.label,
            "converged": self.converged,
            "message": self.message,
        }
        if self.critical_surface is not None:
            for key, value in self.critical_surface.summary_dict().items():
                data[f"critical_{key}"] = value
        return data


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

    This is a practical limit-equilibrium solver for batch data generation and
    screening. It uses circular slip-surface search with vertical slices and the
    Morgenstern-Price-style half-sine interslice-force shape as the search
    convention; the factor of safety is solved with a stable Bishop fixed-point
    form for homogeneous circular surfaces.
    """

    validation_error = _validate_slope(slope)
    if validation_error:
        return SafetyResult(
            fos=float("nan"),
            label="invalid",
            converged=False,
            message=validation_error,
        )

    candidates: list[SlipSurface] = []
    for surface in _candidate_surfaces(slope):
        solved = _solve_surface(
            slope,
            surface,
            include_points=include_points,
            include_slices=include_slices,
        )
        if solved is not None and isfinite(solved.fos) and solved.fos > 0:
            candidates.append(solved)

    if not candidates:
        return SafetyResult(
            fos=float("nan"),
            label="invalid",
            converged=False,
            message="No valid slip surface found for these parameters.",
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
    )


def result_to_dict(result: SafetyResult) -> dict:
    data = result.to_flat_dict()
    data["candidates"] = [surface.summary_dict() for surface in result.candidates]
    if result.critical_surface is not None:
        data["critical_surface"] = result.critical_surface.summary_dict()
        data["critical_slices"] = [asdict(item) for item in result.critical_surface.slices]
    return data


@dataclass(frozen=True)
class _SurfaceSpec:
    center_x: float
    center_y: float
    radius: float
    entry_x: float
    entry_y: float
    sagitta: float


def _validate_slope(slope: SlopeInput) -> str | None:
    checks = {
        "gamma": slope.gamma,
        "c": slope.c,
        "beta": slope.beta,
        "phi": slope.phi,
        "H": slope.H,
        "ru": slope.ru,
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
    return None


def _slope_length(slope: SlopeInput) -> float:
    return slope.H / tan(radians(slope.beta))


def _ground_y(slope: SlopeInput, x: float | np.ndarray) -> float | np.ndarray:
    return np.asarray(x) * tan(radians(slope.beta))


def _candidate_surfaces(slope: SlopeInput) -> Iterable[_SurfaceSpec]:
    length = _slope_length(slope)
    density = int(slope.search_density)
    entry_fracs = np.linspace(0.35, 1.0, density + 2)
    sagitta_fracs = np.linspace(0.08, 0.70, density + 2)

    for entry_frac in entry_fracs:
        entry_x = float(length * entry_frac)
        entry_y = float(_ground_y(slope, entry_x))
        p0 = np.array([0.0, 0.0])
        p1 = np.array([entry_x, entry_y])
        chord = p1 - p0
        chord_len = float(np.linalg.norm(chord))
        if chord_len <= 0:
            continue

        unit = chord / chord_len
        below_normal = np.array([unit[1], -unit[0]])
        midpoint = (p0 + p1) / 2.0

        for sagitta_frac in sagitta_fracs:
            sagitta = min(float(slope.H * sagitta_frac), chord_len * 0.90)
            if sagitta <= 0:
                continue
            radius = chord_len**2 / (8.0 * sagitta) + sagitta / 2.0
            center = midpoint - below_normal * (radius - sagitta)
            yield _SurfaceSpec(
                center_x=float(center[0]),
                center_y=float(center[1]),
                radius=float(radius),
                entry_x=entry_x,
                entry_y=entry_y,
                sagitta=sagitta,
            )


def _solve_surface(
    slope: SlopeInput,
    surface: _SurfaceSpec,
    *,
    include_points: bool,
    include_slices: bool,
) -> SlipSurface | None:
    xs = np.linspace(0.0, surface.entry_x, slope.num_slices + 1)
    mids = (xs[:-1] + xs[1:]) / 2.0
    widths = xs[1:] - xs[:-1]

    base_mid = _circle_y(surface, mids)
    base_left = _circle_y(surface, xs[:-1])
    base_right = _circle_y(surface, xs[1:])
    ground_mid = _ground_y(slope, mids)
    heights = ground_mid - base_mid
    if (
        np.any(~np.isfinite(base_mid))
        or np.any(~np.isfinite(heights))
        or np.any(heights <= 0)
    ):
        return None

    max_depth = float(np.max(ground_mid - base_mid))
    if max_depth > max(slope.H * 3.0, 5.0):
        return None

    alpha = np.abs(np.arctan2(base_right - base_left, widths))
    if np.any(alpha >= radians(88.0)):
        return None

    weights = slope.gamma * heights * widths
    if np.any(weights <= 0):
        return None

    fos = _bishop_fixed_point(
        weights=weights,
        widths=widths,
        alpha=alpha,
        c=slope.c,
        phi_deg=slope.phi,
        ru=slope.ru,
    )
    if not isfinite(fos) or fos <= 0:
        return None

    points: list[tuple[float, float]] = []
    slices: list[SliceData] = []
    if include_points:
        px = np.linspace(0.0, surface.entry_x, 120)
        py = _circle_y(surface, px)
        points = [(float(x), float(y)) for x, y in zip(px, py) if isfinite(float(y))]

    if include_slices:
        slices = [
            SliceData(
                x_left=float(xs[i]),
                x_right=float(xs[i + 1]),
                x_mid=float(mids[i]),
                ground_y=float(ground_mid[i]),
                base_y=float(base_mid[i]),
                height=float(heights[i]),
                alpha_deg=float(degrees(alpha[i])),
                weight=float(weights[i]),
            )
            for i in range(len(mids))
        ]

    return SlipSurface(
        fos=float(fos),
        center_x=surface.center_x,
        center_y=surface.center_y,
        radius=surface.radius,
        entry_x=surface.entry_x,
        entry_y=surface.entry_y,
        sagitta=surface.sagitta,
        points=points,
        slices=slices,
    )


def _circle_y(surface: _SurfaceSpec, x: float | np.ndarray) -> np.ndarray:
    x_array = np.asarray(x, dtype=float)
    radicand = surface.radius**2 - (x_array - surface.center_x) ** 2
    y = np.full_like(x_array, np.nan, dtype=float)
    valid = radicand >= 0.0
    y[valid] = surface.center_y - np.sqrt(radicand[valid])
    return y


def _bishop_fixed_point(
    *,
    weights: np.ndarray,
    widths: np.ndarray,
    alpha: np.ndarray,
    c: float,
    phi_deg: float,
    ru: float,
    max_iter: int = 100,
    tol: float = 1e-5,
) -> float:
    phi_tan = tan(radians(phi_deg))
    driving = float(np.sum(weights * np.sin(alpha)))
    if driving <= 1e-9:
        return float("nan")

    pore_reduction = np.clip(ru, 0.0, 0.95)
    effective_weight = weights * (1.0 - pore_reduction)
    strength = c * widths + effective_weight * phi_tan

    fos = 1.2
    for _ in range(max_iter):
        denominator = np.cos(alpha) + (np.sin(alpha) * phi_tan / max(fos, 1e-6))
        if np.any(denominator <= 1e-9):
            return float("nan")
        updated = float(np.sum(strength / denominator) / driving)
        if not isfinite(updated) or updated <= 0:
            return float("nan")
        if abs(updated - fos) < tol:
            return updated
        fos = 0.5 * fos + 0.5 * updated
    return fos

