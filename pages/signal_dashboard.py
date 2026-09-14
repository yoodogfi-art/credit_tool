"""Page: Signal Dashboard — entity x entity spread / Z-score matrix."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from assets.styles import DEEP_GREEN, HEATMAP_DIVERG
from data.loader import TENOR_LABELS, POLICY_RATE_SECTOR
from chart_utils import PLOTLY_CONFIG, date_range_picker

_RATING_ORDER = [
    "AAA", "AA+", "AA", "AA-", "A+", "A", "A-",
    "BBB+", "BBB", "BBB-", "BB+", "BB", "BB-",
    "B+", "B", "B-", "CCC+", "CCC", "CCC-", "CC", "C", "D",
]


def _rating_sort_key(r: str) -> tuple:
    try:
        return (0, _RATING_ORDER.index(r))
    except ValueError:
        return (1, r)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def _series(df: pd.DataFrame, cat: str, tenor: str) -> pd.Series:
    s = df[(df["category"] == cat) & (df["tenor"] == tenor)]
    return s.set_index("date")["yield"].sort_index().dropna()


def _rolling_z(spread: pd.Series, window: int) -> pd.Series:
    min_periods = max(5, window // 3)
    mean = spread.rolling(window, min_periods=min_periods).mean()
    std = spread.rolling(window, min_periods=min_periods).std(ddof=0)
    z = (spread - mean) / std
    return z.replace([np.inf, -np.inf], np.nan)


def _spread_and_z(a: pd.Series, b: pd.Series, window: int) -> tuple[float, float]:
    """Return (latest spread bp, latest rolling z) for (b - a)."""
    idx = a.index.intersection(b.index)
    if len(idx) < 2:
        return np.nan, np.nan
    spread = ((b - a) * 100).reindex(idx).dropna()
    if spread.empty:
        return np.nan, np.nan
    z = _rolling_z(spread, window).dropna()
    return spread.iloc[-1], (z.iloc[-1] if len(z) else np.nan)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def render(df: pd.DataFrame) -> None:
    st.header("Signal Dashboard")
    st.caption("전체 계열 간 스프레드 / Z-score 매트릭스 (상관관계 매트릭스와 동일한 구성)")

    st.markdown("**분석 기간**")
    d_start, d_end = date_range_picker(df, "sig")
    dff = df[df["date"].between(d_start, d_end)]
    if dff.empty:
        st.warning("선택한 기간에 데이터가 없습니다.")
        return

    c1, c2, c3 = st.columns(3)
    tenor = c1.selectbox("만기", TENOR_LABELS,
                          index=TENOR_LABELS.index("3Y") if "3Y" in TENOR_LABELS else 0,
                          key="sig_tenor")
    z_window = c2.number_input("Z-score 산출기간 (영업일)", value=60, min_value=10, max_value=500, key="sig_zwin")
    z_thresh = c3.number_input("Z-score 임계값", value=2.0, min_value=0.5, max_value=5.0, step=0.1, key="sig_zthresh")

    all_sectors = sorted(s for s in dff["sector"].unique() if s != POLICY_RATE_SECTOR)
    if not all_sectors:
        st.warning("표시할 계열이 없습니다.")
        return

    sel_sectors = st.multiselect(
        "표시할 섹터 (해제하면 매트릭스에서 제외)", all_sectors,
        default=all_sectors, key="sig_sectors",
    )
    if not sel_sectors:
        st.info("표시할 섹터를 하나 이상 선택하세요.")
        return

    tdf = dff[(dff["tenor"] == tenor) & (dff["sector"].isin(sel_sectors))]
    if tdf.empty:
        st.warning(f"선택한 섹터에 {tenor} 데이터가 없습니다.")
        return

    cat_info = tdf[["category", "sector", "rating"]].drop_duplicates().set_index("category")
    sector_rank = {s: i for i, s in enumerate(sel_sectors)}

    def _entity_key(cat: str) -> tuple:
        info = cat_info.loc[cat]
        return (sector_rank.get(info["sector"], 999), _rating_sort_key(info["rating"]))

    entities = sorted(cat_info.index.unique().tolist(), key=_entity_key)
    n = len(entities)
    if n < 2:
        st.warning("비교할 계열이 2개 이상 필요합니다.")
        return
    if n > 30:
        st.caption(f"⚠ {n}개 계열이 선택되어 매트릭스가 큽니다. 섹터를 줄이면 더 보기 편합니다.")

    series_map = {cat: _series(dff, cat, tenor) for cat in entities}

    z_mat = np.full((n, n), np.nan)
    sp_mat = np.full((n, n), np.nan)
    text = [["" for _ in range(n)] for _ in range(n)]
    hover = [["" for _ in range(n)] for _ in range(n)]

    breach_n, pair_n = 0, 0
    for i in range(n):
        z_mat[i, i] = 0.0
        sp_mat[i, i] = 0.0
        hover[i][i] = entities[i]
        for j in range(i + 1, n):
            sp_now, z_now = _spread_and_z(series_map[entities[i]], series_map[entities[j]], int(z_window))
            sp_mat[i, j], sp_mat[j, i] = sp_now, (-sp_now if not np.isnan(sp_now) else np.nan)
            z_mat[i, j], z_mat[j, i] = z_now, (-z_now if not np.isnan(z_now) else np.nan)

            if not np.isnan(sp_now):
                text[i][j] = f"{sp_now:+.0f}bp<br>Z{z_now:+.1f}"
                text[j][i] = f"{-sp_now:+.0f}bp<br>Z{-z_now:+.1f}"
                hover[i][j] = f"{entities[j]} − {entities[i]}<br>스프레드: {sp_now:+.1f}bp<br>Z-score: {z_now:+.2f}"
                hover[j][i] = f"{entities[i]} − {entities[j]}<br>스프레드: {-sp_now:+.1f}bp<br>Z-score: {-z_now:+.2f}"
                pair_n += 1
                if abs(z_now) >= z_thresh:
                    breach_n += 1
            else:
                hover[i][j] = f"{entities[j]} vs {entities[i]}: 데이터 없음"
                hover[j][i] = f"{entities[i]} vs {entities[j]}: 데이터 없음"

    st.markdown(
        f'<div style="background:#F7F8F5;border-radius:6px;padding:14px 18px;'
        f'border-left:4px solid {DEEP_GREEN};margin:12px 0">'
        f'<div style="font-size:11px;color:#888;margin-bottom:4px">종합</div>'
        f'<div style="font-size:14px;font-weight:600;color:{DEEP_GREEN}">'
        f'{pair_n}개 쌍 중 {breach_n}개 임계값(±{z_thresh}) 초과 &nbsp;|&nbsp; '
        f'{n}개 계열 x {tenor}</div></div>',
        unsafe_allow_html=True,
    )
    st.caption("셀 = (열 계열) − (행 계열) 스프레드. 색상은 Z-score(±3 기준), 텍스트는 스프레드(bp)와 Z-score.")

    fig = go.Figure(go.Heatmap(
        z=z_mat.tolist(), x=entities, y=entities,
        text=text, texttemplate="%{text}",
        hovertext=hover, hoverinfo="text",
        colorscale=HEATMAP_DIVERG, zmid=0, zmin=-3, zmax=3, showscale=True,
        colorbar=dict(title=dict(text="Z", side="right"), thickness=12, len=0.8),
        textfont=dict(size=max(7, 11 - n // 6)),
    ))
    fig.update_layout(
        height=max(400, n * 34 + 60),
        margin=dict(l=140, r=30, t=10, b=10),
        font=dict(family="Apple SD Gothic Neo, Noto Sans KR, sans-serif", size=10),
        xaxis=dict(side="top", tickangle=-45),
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
