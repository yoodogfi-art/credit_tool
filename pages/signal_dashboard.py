"""Page: Signal Dashboard — entity x entity and sector x sector spread / Z-score matrices."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from assets.styles import DEEP_GREEN, HEATMAP_DIVERG
from data.loader import TENOR_LABELS, POLICY_RATE_SECTOR
from chart_utils import PLOTLY_CONFIG, date_range_picker, sector_tenor_picker

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


def _sector_series(df: pd.DataFrame, sector: str, tenor: str) -> pd.Series:
    """Average yield across every rating within a sector, by date (rating ignored)."""
    s = df[(df["sector"] == sector) & (df["tenor"] == tenor)]
    if s.empty:
        return pd.Series(dtype=float)
    return s.groupby("date")["yield"].mean().sort_index()


def entities_from_pairs(dff: pd.DataFrame, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    """Build one row per (category, tenor) present for each checked (sector, tenor) pair."""
    rows = []
    for sector, tenor in pairs:
        sub = dff[(dff["sector"] == sector) & (dff["tenor"] == tenor)]
        if sub.empty:
            continue
        for cat, rating in sub[["category", "rating"]].drop_duplicates().itertuples(index=False):
            rows.append({"label": f"{cat} {tenor}", "category": cat, "tenor": tenor,
                         "sector": sector, "rating": rating})
    cols = ["label", "category", "tenor", "sector", "rating"]
    if not rows:
        return pd.DataFrame(columns=cols).set_index("label")
    return pd.DataFrame(rows, columns=cols).drop_duplicates(subset="label").set_index("label")


def _rolling_z(spread: pd.Series, window: int) -> pd.Series:
    # A small, mostly-fixed floor (rather than a fraction of the window) so
    # pairs with a shorter overlap than the full window still get a Z-score
    # instead of going blank — any real overlap should produce a number.
    min_periods = min(len(spread), max(5, min(10, window)))
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
    if len(spread) < 2:
        return (spread.iloc[-1] if len(spread) else np.nan), np.nan
    z = _rolling_z(spread, window).dropna()
    return spread.iloc[-1], (z.iloc[-1] if len(z) else np.nan)


def _pairwise_matrix(names: list[str], series_map: dict[str, pd.Series], window: int, z_thresh: float):
    """Full symmetric (sign-flipped) spread/Z matrix across every name pair."""
    n = len(names)
    z_mat = np.full((n, n), np.nan)
    sp_mat = np.full((n, n), np.nan)
    text = [["" for _ in range(n)] for _ in range(n)]
    hover = [["" for _ in range(n)] for _ in range(n)]

    breach_n, pair_n = 0, 0
    for i in range(n):
        z_mat[i, i] = 0.0
        sp_mat[i, i] = 0.0
        hover[i][i] = names[i]
        for j in range(i + 1, n):
            sp_now, z_now = _spread_and_z(series_map[names[i]], series_map[names[j]], window)
            sp_mat[i, j] = sp_now
            sp_mat[j, i] = -sp_now if not np.isnan(sp_now) else np.nan
            z_mat[i, j] = z_now
            z_mat[j, i] = -z_now if not np.isnan(z_now) else np.nan

            if not np.isnan(sp_now):
                z_txt = f"Z{z_now:+.1f}" if not np.isnan(z_now) else "Z-"
                z_txt_neg = f"Z{-z_now:+.1f}" if not np.isnan(z_now) else "Z-"
                text[i][j] = f"{sp_now:+.0f}bp<br>{z_txt}"
                text[j][i] = f"{-sp_now:+.0f}bp<br>{z_txt_neg}"
                z_str = f"{z_now:+.2f}" if not np.isnan(z_now) else "-"
                z_str_neg = f"{-z_now:+.2f}" if not np.isnan(z_now) else "-"
                hover[i][j] = f"{names[j]} − {names[i]}<br>스프레드: {sp_now:+.1f}bp<br>Z-score: {z_str}"
                hover[j][i] = f"{names[i]} − {names[j]}<br>스프레드: {-sp_now:+.1f}bp<br>Z-score: {z_str_neg}"
                pair_n += 1
                if not np.isnan(z_now) and abs(z_now) >= z_thresh:
                    breach_n += 1
            else:
                hover[i][j] = f"{names[j]} vs {names[i]}: 데이터 없음"
                hover[j][i] = f"{names[i]} vs {names[j]}: 데이터 없음"

    return z_mat, sp_mat, text, hover, breach_n, pair_n


def _render_heatmap(names: list[str], z_mat: np.ndarray, text: list, hover: list) -> None:
    n = len(names)
    fig = go.Figure(go.Heatmap(
        z=z_mat.tolist(), x=names, y=names,
        text=text, texttemplate="%{text}",
        hovertext=hover, hoverinfo="text",
        colorscale=HEATMAP_DIVERG, zmid=0, zmin=-3, zmax=3, showscale=True,
        colorbar=dict(title=dict(text="Z", side="right"), thickness=12, len=0.8),
        textfont=dict(size=max(7, 11 - n // 6)),
    ))
    fig.update_layout(
        height=max(320, n * 34 + 60),
        margin=dict(l=140, r=30, t=10, b=10),
        font=dict(family="Apple SD Gothic Neo, Noto Sans KR, sans-serif", size=10),
        xaxis=dict(side="top", tickangle=-45),
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)


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

    c1, c2 = st.columns(2)
    z_window = c1.number_input("Z-score 산출기간 (영업일)", value=60, min_value=10, max_value=500, key="sig_zwin")
    z_thresh = c2.number_input("Z-score 임계값", value=2.0, min_value=0.5, max_value=5.0, step=0.1, key="sig_zthresh")

    all_sectors = sorted(s for s in dff["sector"].unique() if s != POLICY_RATE_SECTOR)
    if not all_sectors:
        st.warning("표시할 계열이 없습니다.")
        return

    pairs = sector_tenor_picker(all_sectors, TENOR_LABELS, "sig", default="single",
                                 default_tenor="3Y" if "3Y" in TENOR_LABELS else None)
    if not pairs:
        st.info("표시할 섹터 x 만기 조합을 하나 이상 선택하세요.")
        return

    # -- Entity (category x maturity) matrix ---------------------------------
    ent_df = entities_from_pairs(dff, pairs)
    if ent_df.empty:
        st.warning("선택한 조합에 데이터가 없습니다.")
        return

    sector_rank = {s: i for i, s in enumerate(all_sectors)}
    tenor_rank = {t: i for i, t in enumerate(TENOR_LABELS)}

    def _entity_key(label: str) -> tuple:
        row = ent_df.loc[label]
        return (sector_rank.get(row["sector"], 999), tenor_rank.get(row["tenor"], 999),
                _rating_sort_key(row["rating"]))

    entities = sorted(ent_df.index.tolist(), key=_entity_key)
    if len(entities) < 2:
        st.warning("비교할 계열이 2개 이상 필요합니다.")
        return
    if len(entities) > 30:
        st.caption(f"⚠ {len(entities)}개 계열이 선택되어 매트릭스가 큽니다. 조합을 줄이면 더 보기 편합니다.")

    entity_series = {lbl: _series(dff, ent_df.loc[lbl, "category"], ent_df.loc[lbl, "tenor"]) for lbl in entities}
    z_mat, sp_mat, text, hover, breach_n, pair_n = _pairwise_matrix(entities, entity_series, int(z_window), z_thresh)

    st.markdown(
        f'<div style="background:#F7F8F5;border-radius:6px;padding:14px 18px;'
        f'border-left:4px solid {DEEP_GREEN};margin:12px 0">'
        f'<div style="font-size:11px;color:#888;margin-bottom:4px">종합</div>'
        f'<div style="font-size:14px;font-weight:600;color:{DEEP_GREEN}">'
        f'{pair_n}개 쌍 중 {breach_n}개 임계값(±{z_thresh}) 초과 &nbsp;|&nbsp; '
        f'{len(entities)}개 계열 ({len(pairs)}개 섹터 x 만기 조합)</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown("#### 계열(섹터 x 등급 x 만기) 매트릭스")
    st.caption("셀 = (열 계열) − (행 계열) 스프레드. 색상은 Z-score(±3 기준), 텍스트는 스프레드(bp)와 Z-score.")
    _render_heatmap(entities, z_mat, text, hover)

    # -- Sector-only matrix (rating ignored) ---------------------------------
    st.markdown("---")
    st.markdown("#### 섹터 매트릭스 (등급 무시, 섹터 내 평균)")
    st.caption("등급을 무시하고 섹터 내 모든 계열의 평균 금리로 계산한, 선택한 섹터 x 만기 조합 간 상대 스프레드입니다.")

    sec_rows = []
    for sector, tenor in pairs:
        ser = _sector_series(dff, sector, tenor)
        if not ser.empty:
            sec_rows.append((sector, tenor, f"{sector} {tenor}"))
    sec_rows.sort(key=lambda r: (sector_rank.get(r[0], 999), tenor_rank.get(r[1], 999)))
    sector_labels = [r[2] for r in sec_rows]
    sector_series_map = {r[2]: _sector_series(dff, r[0], r[1]) for r in sec_rows}

    if len(sector_labels) < 2:
        st.info("섹터 매트릭스를 표시하려면 유효한 섹터 x 만기 조합이 2개 이상 필요합니다.")
    else:
        zs_mat, sps_mat, text_s, hover_s, breach_s, pair_s = _pairwise_matrix(
            sector_labels, sector_series_map, int(z_window), z_thresh
        )
        st.caption(f"{pair_s}개 쌍 중 {breach_s}개 임계값(±{z_thresh}) 초과")
        _render_heatmap(sector_labels, zs_mat, text_s, hover_s)
