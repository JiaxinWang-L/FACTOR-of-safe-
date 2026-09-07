from __future__ import annotations

from io import StringIO
from math import radians, tan
from pathlib import Path
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
SRC_DIR = APP_DIR / "src"
for import_path in (APP_DIR, SRC_DIR):
    path_text = str(import_path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

try:
    from src.morgenstern_price import SlopeInput, SafetyResult, compute_factor_of_safety
except ModuleNotFoundError:
    from morgenstern_price import SlopeInput, SafetyResult, compute_factor_of_safety


st.set_page_config(page_title="安全系数计算 APP", layout="wide")


def ground_y(slope: SlopeInput, x: float) -> float:
    return x * tan(radians(slope.beta))


def build_figure(
    slope: SlopeInput,
    result: SafetyResult,
    *,
    show_slices: bool,
    max_candidates: int,
) -> go.Figure:
    length = slope.H / tan(radians(slope.beta))
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=[0, length, length, 0, 0],
            y=[0, slope.H, 0, 0, 0],
            fill="toself",
            mode="lines",
            line=dict(color="#8B7355", width=2),
            fillcolor="rgba(196, 164, 132, 0.35)",
            name="边坡土体",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, length],
            y=[0, slope.H],
            mode="lines+markers",
            line=dict(color="#3A3A3A", width=3),
            marker=dict(size=8),
            name="坡面",
            hovertemplate="x=%{x:.2f} m<br>y=%{y:.2f} m<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, length],
            y=[0, 0],
            mode="lines",
            line=dict(color="#4C4C4C", width=1, dash="dash"),
            name="基准线",
            hoverinfo="skip",
        )
    )

    for rank, surface in enumerate(result.candidates[:max_candidates], start=1):
        if not surface.points:
            continue
        x_values = [point[0] for point in surface.points]
        y_values = [point[1] for point in surface.points]
        opacity = max(0.18, 0.75 - rank * 0.045)
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="lines",
                line=dict(color=f"rgba(37, 99, 235, {opacity})", width=1.6),
                name=f"候选滑面 {rank}",
                hovertemplate=(
                    f"候选滑面 {rank}<br>"
                    f"FOS={surface.fos:.3f}<br>"
                    f"圆心=({surface.center_x:.2f}, {surface.center_y:.2f}) m<br>"
                    f"半径={surface.radius:.2f} m<extra></extra>"
                ),
            )
        )

    critical = result.critical_surface
    if critical is not None and critical.points:
        fig.add_trace(
            go.Scatter(
                x=[point[0] for point in critical.points],
                y=[point[1] for point in critical.points],
                mode="lines",
                line=dict(color="#DC2626", width=5),
                name="临界滑动面",
                hovertemplate=(
                    f"临界滑动面<br>FOS={critical.fos:.3f}<br>"
                    f"圆心=({critical.center_x:.2f}, {critical.center_y:.2f}) m<br>"
                    f"半径={critical.radius:.2f} m<extra></extra>"
                ),
            )
        )

        if show_slices:
            for item in critical.slices:
                fig.add_trace(
                    go.Scatter(
                        x=[item.x_mid, item.x_mid],
                        y=[item.base_y, item.ground_y],
                        mode="lines",
                        line=dict(color="rgba(70, 70, 70, 0.45)", width=1),
                        showlegend=False,
                        hovertemplate=(
                            f"条分<br>x={item.x_mid:.2f} m<br>"
                            f"高度={item.height:.2f} m<br>"
                            f"重量={item.weight:.2f} kN/m<extra></extra>"
                        ),
                    )
                )

    fig.add_annotation(x=0, y=0, text="坡脚", showarrow=True, ax=-30, ay=30)
    fig.add_annotation(x=length, y=slope.H, text="坡顶", showarrow=True, ax=30, ay=-30)
    fig.update_layout(
        height=650,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis_title="水平距离 x (m)",
        yaxis_title="高程 y (m)",
        template="plotly_white",
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(range=[-0.08 * length, 1.12 * length])
    fig.update_yaxes(range=[min(-0.35 * slope.H, -2), 1.22 * slope.H])
    return fig


def candidate_table(result: SafetyResult, max_candidates: int) -> pd.DataFrame:
    rows = []
    for rank, surface in enumerate(result.candidates[:max_candidates], start=1):
        rows.append(
            {
                "rank": rank,
                "FOS": surface.fos,
                "center_x": surface.center_x,
                "center_y": surface.center_y,
                "radius": surface.radius,
                "entry_x": surface.entry_x,
                "entry_y": surface.entry_y,
            }
        )
    return pd.DataFrame(rows)


def result_csv(slope: SlopeInput, result: SafetyResult) -> str:
    row = {
        "gamma": slope.gamma,
        "c": slope.c,
        "beta": slope.beta,
        "phi": slope.phi,
        "H": slope.H,
        "ru": slope.ru,
        "num_slices": slope.num_slices,
        "search_density": slope.search_density,
        **result.to_flat_dict(),
    }
    buffer = StringIO()
    pd.DataFrame([row]).to_csv(buffer, index=False, encoding="utf-8-sig")
    return buffer.getvalue()


st.title("安全系数计算 APP")
st.caption("简单均质边坡 · 圆弧滑动面搜索 · 摩根斯坦-普莱斯风格条分计算")

with st.sidebar:
    st.header("边坡参数")
    gamma = st.number_input("单位重 gamma (kN/m³)", min_value=1.0, max_value=40.0, value=18.0, step=0.1)
    c = st.number_input("黏聚力 c (kPa)", min_value=0.0, max_value=250.0, value=20.0, step=0.5)
    beta = st.number_input("坡角 beta (°)", min_value=3.0, max_value=80.0, value=35.0, step=0.5)
    phi = st.number_input("内摩擦角 phi (°)", min_value=0.0, max_value=70.0, value=28.0, step=0.5)
    height = st.number_input("坡高 H (m)", min_value=0.5, max_value=500.0, value=20.0, step=0.5)
    ru = st.slider("孔压系数 ru", min_value=0.0, max_value=0.95, value=0.0, step=0.01)

    st.header("计算与显示")
    num_slices = st.slider("条分数量", min_value=12, max_value=80, value=30, step=2)
    search_density = st.slider("滑面搜索精度", min_value=3, max_value=14, value=8, step=1)
    max_candidates = st.slider("候选滑面显示数量", min_value=1, max_value=20, value=10, step=1)
    show_slices = st.checkbox("显示条分线", value=True)
    run = st.button("计算安全系数", type="primary", use_container_width=True)

if run:
    slope = SlopeInput(
        gamma=gamma,
        c=c,
        beta=beta,
        phi=phi,
        H=height,
        ru=ru,
        num_slices=num_slices,
        search_density=search_density,
    )
    result = compute_factor_of_safety(
        slope,
        candidate_limit=max_candidates,
        include_points=True,
        include_slices=True,
    )

    if not result.converged:
        st.error(result.message)
        st.stop()

    metric_cols = st.columns(4)
    metric_cols[0].metric("安全系数 FOS", f"{result.fos:.3f}")
    metric_cols[1].metric("稳定性标签", result.label)
    metric_cols[2].metric("候选滑面数", len(result.candidates))
    metric_cols[3].metric("孔压系数 ru", f"{slope.ru:.2f}")

    if result.label == "stable":
        st.success("当前参数下边坡为 stable。")
    elif result.label == "critical":
        st.warning("当前参数下边坡为 critical。")
    else:
        st.error("当前参数下边坡为 unstable。")

    fig = build_figure(
        slope,
        result,
        show_slices=show_slices,
        max_candidates=max_candidates,
    )
    st.plotly_chart(fig, use_container_width=True)

    table = candidate_table(result, max_candidates)
    st.subheader("候选滑动面结果")
    st.dataframe(table, use_container_width=True, hide_index=True)

    st.download_button(
        "下载当前计算结果 CSV",
        data=result_csv(slope, result),
        file_name="single_slope_safety_result.csv",
        mime="text/csv",
        use_container_width=True,
    )
else:
    st.info("在左侧输入边坡参数，然后点击“计算安全系数”。")
