"""Page: Signal Dashboard — base vs. comparison group spread Z-score."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from assets.styles import CHART_COLORS, CORAL, DEEP_GREEN
from data.loader import TENOR_LABELS
from chart_utils import PLOTLY_CONFIG, date_range_picker


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def _matching_categories(df: pd.DataFrame, sectors: list[str], ratings: list[str]) -> list[str]:
    if not sectors or not ratings:
        return []
    sub = df[df["sector"].isin(sectors) & df["rating"].isin(ratings)]
    return sorted(sub["category"].unique().tolist())


def _group_series(df: pd.DataFrame, categories: list[str], tenor: str) -> pd.Series:
    """Average yield across categories, by date, for one tenor."""
    sub = df[df["category"].isin(categories) & (df["tenor"] == tenor)]
    if sub.empty:
        return pd.Series(dtype=float)
    return sub.groupby("date")["yield"].mean().sort_index()


def _rolling_z(spread: pd.Series, window: int) -> pd.Series:
    min_periods = max(5, window // 3)
    mean = spread.rolling(window, min_periods=min_periods).mean()
    std = spread.rolling(window, min_periods=min_periods).std(ddof=0)
    z = (spread - mean) / std
    return z.replace([np.inf, -np.inf], np.nan)


def _spread_and_z(base: pd.Series, cmp_series: pd.Series, window: int) -> dict:
    idx = base.index.intersection(cmp_series.index)
    if len(idx) < 2:
        return {"spread": pd.Series(dtype=float), "z": pd.Series(dtype=float),
                "current": np.nan, "mean": np.nan, "std": np.nan, "z_now": np.nan}
    spread = ((cmp_series - base) * 100).reindex(idx).dropna()
    z = _rolling_z(spread, window)
    z_valid = z.dropna()
    win = spread.iloc[-window:] if len(spread) > window else spread
    return {
        "spread": spread,
        "z": z,
        "current": spread.iloc[-1] if len(spread) else np.nan,
        "mean": win.mean() if len(win) else np.nan,
        "std": win.std(ddof=0) if len(win) else np.nan,
        "z_now": z_valid.iloc[-1] if len(z_valid) else np.nan,
    }


def _status(z_now: float, thresh: float) -> tuple[str, str]:
    if np.isnan(z_now):
        return "데이터 부족", "#9E9E9E"
    if z_now >= thresh:
        return "스프레드 확대 (임계값 초과)", CORAL
    if z_now <= -thresh:
        return "스프레드 축소 (임계값 초과)", "#1B5E20"
    return "정상 범위", "#616161"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def render(df: pd.DataFrame) -> None:
    st.header("Signal Dashboard")
    st.caption("Base 그룹 대비 비교 그룹의 스프레드 Z-score 시그널")

    all_sectors = sorted(df["sector"].unique().tolist())
    all_ratings = sorted([r for r in df["rating"].unique().tolist() if r])

    st.markdown("**분석 기간**")
    d_start, d_end = date_range_picker(df, "sig")
    dff = df[df["date"].between(d_start, d_end)]
    if dff.empty:
        st.warning("선택한 기간에 데이터가 없습니다.")
        return

    tenor_default = TENOR_LABELS.index("3Y") if "3Y" in TENOR_LABELS else 0
    tenor = st.selectbox("만기", TENOR_LABELS, index=tenor_default, key="sig_tenor")

    st.markdown("---")
    c_base, c_cmp = st.columns(2)
    with c_base:
        st.markdown("#### Base 그룹")
        base_sectors = st.multiselect("섹터 (Base)", all_sectors, key="sig_base_sectors")
        base_ratings = st.multiselect("등급 (Base)", all_ratings, key="sig_base_ratings")
        base_cats = _matching_categories(dff, base_sectors, base_ratings)
        if base_cats:
            st.caption(f"구성 계열 ({len(base_cats)}개, 평균): {', '.join(base_cats)}")
    with c_cmp:
        st.markdown("#### 비교 그룹")
        cmp_sectors = st.multiselect("섹터 (비교)", all_sectors, key="sig_cmp_sectors")
        cmp_ratings = st.multiselect("등급 (비교)", all_ratings, key="sig_cmp_ratings")
        cmp_cats = _matching_categories(dff, cmp_sectors, cmp_ratings)
        if cmp_cats:
            st.caption(f"구성 계열 ({len(cmp_cats)}개, 개별 비교): {', '.join(cmp_cats)}")

    if not base_cats or not cmp_cats:
        st.info("Base 그룹과 비교 그룹의 섹터/등급을 각각 하나 이상 선택하세요.")
        return

    z1, z2 = st.columns(2)
    z_window = z1.number_input("Z-score 산출기간 (영업일)", value=60, min_value=10, max_value=500, key="sig_zwin")
    z_thresh = z2.number_input("Z-score 임계값", value=2.0, min_value=0.5, max_value=5.0, step=0.1, key="sig_zthresh")

    base_series = _group_series(dff, base_cats, tenor)
    if base_series.empty:
        st.warning(f"Base 그룹의 {tenor} 데이터가 없습니다.")
        return

    results: dict[str, dict] = {}
    for cat in cmp_cats:
        cat_series = _group_series(dff, [cat], tenor)
        if cat_series.empty:
            continue
        results[cat] = _spread_and_z(base_series, cat_series, int(z_window))

    if not results:
        st.warning("비교 그룹의 데이터가 없습니다.")
        return

    breach_n = sum(1 for r in results.values() if not np.isnan(r["z_now"]) and abs(r["z_now"]) >= z_thresh)
    st.markdown(
        f'<div style="background:#F7F8F5;border-radius:6px;padding:14px 18px;'
        f'border-left:4px solid {DEEP_GREEN};margin:12px 0">'
        f'<div style="font-size:11px;color:#888;margin-bottom:4px">종합</div>'
        f'<div style="font-size:14px;font-weight:600;color:{DEEP_GREEN}">'
        f'{len(results)}개 비교 계열 중 {breach_n}개 임계값(±{z_thresh}) 초과</div></div>',
        unsafe_allow_html=True,
    )

    # Detail table
    st.markdown("#### 상세")
    rows_html = (
        '<style>.sigt{border-collapse:collapse;width:100%;font-size:12px;'
        "font-family:'Apple SD Gothic Neo',sans-serif}"
        '.sigt th{background:#2D3F38;color:#fff;padding:8px 12px;text-align:left}'
        '.sigt td{padding:7px 12px;border-bottom:1px solid #EEEEEE}'
        '.sigt tr:hover td{background:#F7F8F5}'
        '</style>'
        '<table class="sigt"><thead><tr>'
        "<th>비교 계열</th><th>현재 스프레드</th><th>평균</th><th>표준편차</th>"
        "<th>Z-score</th><th>상태</th>"
        "</tr></thead><tbody>"
    )
    for cat, r in results.items():
        status_label, status_color = _status(r["z_now"], z_thresh)
        fmt = lambda v, suf="bp": f"{v:.1f}{suf}" if not np.isnan(v) else "-"
        z_str = f"{r['z_now']:+.2f}" if not np.isnan(r["z_now"]) else "-"
        rows_html += (
            f'<tr><td>{cat}</td><td>{fmt(r["current"])}</td><td>{fmt(r["mean"])}</td>'
            f'<td>{fmt(r["std"])}</td><td>{z_str}</td>'
            f'<td style="color:{status_color};font-weight:700">{status_label}</td></tr>'
        )
    rows_html += "</tbody></table>"
    st.markdown(rows_html, unsafe_allow_html=True)

    # Z-score chart
    st.markdown("---")
    st.markdown("#### Z-score 추이")
    fig = go.Figure()
    for i, (cat, r) in enumerate(results.items()):
        z = r["z"].dropna()
        if z.empty:
            continue
        fig.add_trace(go.Scatter(
            x=z.index, y=z.values, name=cat,
            line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=2),
            hovertemplate=f"{cat}: %{{y:.2f}}<extra></extra>",
        ))
    fig.add_hline(y=z_thresh, line_dash="dash", line_color=CORAL, line_width=1,
                  annotation_text=f"+{z_thresh}", annotation_position="top left")
    fig.add_hline(y=-z_thresh, line_dash="dash", line_color=CORAL, line_width=1,
                  annotation_text=f"-{z_thresh}", annotation_position="bottom left")
    fig.add_hline(y=0, line_color="#CCCCCC", line_width=1)
    fig.update_layout(
        template="plotly_white", height=320,
        title=dict(text=f"<b>스프레드 Z-score (base: {', '.join(base_cats)} | {tenor})</b>",
                   font=dict(color=DEEP_GREEN, size=13), x=0),
        font=dict(family="Apple SD Gothic Neo, Noto Sans KR, sans-serif", size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=30, t=48, b=40),
        plot_bgcolor="white", paper_bgcolor="white",
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    # Raw spread chart
    with st.expander("스프레드 원시값 (bp)", expanded=False):
        fig2 = go.Figure()
        for i, (cat, r) in enumerate(results.items()):
            s = r["spread"]
            if s.empty:
                continue
            fig2.add_trace(go.Scatter(
                x=s.index, y=s.values, name=cat,
                line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=2),
                hovertemplate=f"{cat}: %{{y:.1f}}bp<extra></extra>",
            ))
        fig2.update_layout(
            template="plotly_white", height=300,
            title=dict(text=f"<b>스프레드 vs {', '.join(base_cats)} ({tenor})</b>",
                       font=dict(color=DEEP_GREEN, size=13), x=0),
            font=dict(family="Apple SD Gothic Neo, Noto Sans KR, sans-serif", size=12),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=50, r=30, t=48, b=40),
            plot_bgcolor="white", paper_bgcolor="white",
            hovermode="x unified",
        )
        fig2.update_yaxes(ticksuffix="bp")
        st.plotly_chart(fig2, use_container_width=True, config=PLOTLY_CONFIG)
