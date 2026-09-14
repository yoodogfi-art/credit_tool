"""Page: Signal Dashboard — entity x entity and sector x sector spread / Z-score matrices."""

import copy

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from assets.styles import DEEP_GREEN, HEATMAP_DIVERG
from data.loader import TENOR_LABELS, POLICY_RATE_SECTOR
from chart_utils import PLOTLY_CONFIG, date_range_picker, category_tenor_picker

_RATING_ORDER = [
    "AAA", "AA+", "AA", "AA-", "A+", "A", "A-",
    "BBB+", "BBB", "BBB-", "BB+", "BB", "BB-",
    "B+", "B", "B-", "CCC+", "CCC", "CCC-", "CC", "C", "D",
]

TENOR_YEARS = dict(zip(TENOR_LABELS, [0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 4, 5]))


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


def entities_from_pairs(dff: pd.DataFrame, pairs: list[tuple[str, str]],
                         cat_meta: pd.DataFrame) -> pd.DataFrame:
    """Build one row per checked (category, tenor) pair that has data."""
    rows = []
    for cat, tenor in pairs:
        if cat not in cat_meta.index:
            continue
        if dff[(dff["category"] == cat) & (dff["tenor"] == tenor)].empty:
            continue
        info = cat_meta.loc[cat]
        rows.append({"label": f"{cat} {tenor}", "category": cat, "tenor": tenor,
                     "sector": info["sector"], "rating": info["rating"]})
    cols = ["label", "category", "tenor", "sector", "rating"]
    if not rows:
        return pd.DataFrame(columns=cols).set_index("label")
    return pd.DataFrame(rows, columns=cols).drop_duplicates(subset="label").set_index("label")


def _rolling_z(spread: pd.Series, window: int) -> pd.Series:
    # 2 is the bare statistical minimum for a standard deviation — anything
    # stricter blanks out cells purely because a pair's overlap or a
    # thinly-reported maturity has fewer rows than an arbitrary threshold,
    # even though there's enough data to produce a real (if noisier) number.
    min_periods = min(len(spread), 2)
    mean = spread.rolling(window, min_periods=min_periods).mean()
    std = spread.rolling(window, min_periods=min_periods).std(ddof=0)
    z = (spread - mean) / std
    return z.replace([np.inf, -np.inf], np.nan)


def _spread_and_z(a: pd.Series, b: pd.Series, window: int) -> tuple[float, float]:
    """Return (latest spread bp, latest rolling z) for (b - a)."""
    idx = a.index.intersection(b.index)
    if len(idx) == 0:
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


def macaulay_duration_years(tenor_years: float, yield_pct: float) -> float:
    """Macaulay duration (years) of a par-priced, semiannual-coupon bond.

    No other complexity (calls, floaters, amortization, day-count
    conventions, ...) is modeled — a rough, consistent yardstick for
    comparing spread "richness" across maturities, not a pricing tool.
    """
    if tenor_years is None or np.isnan(tenor_years) or tenor_years <= 0 or np.isnan(yield_pct):
        return np.nan
    m = 2  # semiannual coupons
    n = max(1, round(tenor_years * m))
    y = yield_pct / 100.0
    if y <= 0:
        return n / m  # degenerate case: fall back to time-weighted average maturity
    per_rate = y / m
    coupon = per_rate * 100.0
    periods = np.arange(1, n + 1)
    cash_flows = np.full(n, coupon, dtype=float)
    cash_flows[-1] += 100.0
    disc = (1 + per_rate) ** periods
    pv = cash_flows / disc
    price = pv.sum()
    mac_dur_periods = (periods * pv).sum() / price
    return mac_dur_periods / m


def _latest_yield(dff: pd.DataFrame, cat: str, tenor: str) -> float:
    s = dff[(dff["category"] == cat) & (dff["tenor"] == tenor)].sort_values("date")
    return float(s.iloc[-1]["yield"]) if not s.empty else np.nan


def _render_heatmap(names: list[str], z_mat: np.ndarray, text: list, hover: list) -> None:
    """Render a square Z-score heatmap, degrading cell-text density as the
    matrix grows so numbers never overlap: full "bp + Z" text up to 10
    names, Z-only past that, and color/hover only once it's too dense for
    any text to stay legible. The PNG export is also sized to the matrix
    (the app's default export size is tuned for small line charts, not a
    dense heatmap) so downloaded images stay readable too.
    """
    n = len(names)
    if n <= 10:
        cell_text, font_size = text, 12
    elif n <= 20:
        cell_text = [[f"{v:+.1f}" if not np.isnan(v) else "" for v in row] for row in z_mat]
        font_size = 10
    else:
        cell_text, font_size = None, 10
        st.caption(f"⚠ {n}개 계열로 셀이 조밀해 숫자 표시는 생략했습니다 — 마우스오버로 값을 확인하거나 계열을 줄여보세요.")

    heatmap_kwargs = dict(
        z=z_mat.tolist(), x=names, y=names,
        hovertext=hover, hoverinfo="text",
        colorscale=HEATMAP_DIVERG, zmid=0, zmin=-3, zmax=3, showscale=True,
        colorbar=dict(title=dict(text="Z", side="right"), thickness=12, len=0.8),
    )
    if cell_text is not None:
        heatmap_kwargs.update(text=cell_text, texttemplate="%{text}", textfont=dict(size=font_size))

    fig = go.Figure(go.Heatmap(**heatmap_kwargs))
    cell_px = 60 if n <= 10 else 42
    fig.update_layout(
        height=max(320, n * cell_px + 90),
        margin=dict(l=150, r=30, t=10, b=10),
        font=dict(family="Apple SD Gothic Neo, Noto Sans KR, sans-serif", size=11),
        xaxis=dict(side="top", tickangle=-45),
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="white", paper_bgcolor="white",
    )

    export_cfg = copy.deepcopy(PLOTLY_CONFIG)
    export_cfg["toImageButtonOptions"]["width"] = max(640, n * 90)
    export_cfg["toImageButtonOptions"]["height"] = max(390, n * 90)
    export_cfg["toImageButtonOptions"]["scale"] = 2
    st.plotly_chart(fig, use_container_width=True, config=export_cfg)


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

    bond_df = dff[dff["sector"] != POLICY_RATE_SECTOR]
    all_cats = sorted(bond_df["category"].unique().tolist())
    if not all_cats:
        st.warning("표시할 계열이 없습니다.")
        return
    cat_meta = bond_df[["category", "sector", "rating"]].drop_duplicates().set_index("category")

    all_sectors = sorted(bond_df["sector"].unique())
    sector_rank = {s: i for i, s in enumerate(all_sectors)}
    tenor_rank = {t: i for i, t in enumerate(TENOR_LABELS)}

    pairs = category_tenor_picker(all_cats, TENOR_LABELS, "sig", default="single",
                                   default_tenor="3Y" if "3Y" in TENOR_LABELS else None)
    if not pairs:
        st.info("표시할 계열 x 만기 조합을 하나 이상 선택하세요.")
        return

    # -- Entity (category x maturity) matrix ---------------------------------
    ent_df = entities_from_pairs(dff, pairs, cat_meta)
    if ent_df.empty:
        st.warning("선택한 조합에 데이터가 없습니다.")
        return

    def _entity_key(label: str) -> tuple:
        row = ent_df.loc[label]
        return (sector_rank.get(row["sector"], 999), tenor_rank.get(row["tenor"], 999),
                _rating_sort_key(row["rating"]))

    entities = sorted(ent_df.index.tolist(), key=_entity_key)
    if len(entities) < 2:
        st.warning("비교할 계열이 2개 이상 필요합니다.")
        return

    entity_series = {lbl: _series(dff, ent_df.loc[lbl, "category"], ent_df.loc[lbl, "tenor"]) for lbl in entities}
    z_mat, sp_mat, text, hover, breach_n, pair_n = _pairwise_matrix(entities, entity_series, int(z_window), z_thresh)

    st.markdown(
        f'<div style="background:#F7F8F5;border-radius:6px;padding:14px 18px;'
        f'border-left:4px solid {DEEP_GREEN};margin:12px 0">'
        f'<div style="font-size:11px;color:#888;margin-bottom:4px">종합</div>'
        f'<div style="font-size:14px;font-weight:600;color:{DEEP_GREEN}">'
        f'{pair_n}개 쌍 중 {breach_n}개 임계값(±{z_thresh}) 초과 &nbsp;|&nbsp; '
        f'{len(entities)}개 계열 ({len(pairs)}개 계열 x 만기 조합)</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown("#### 계열(섹터 x 등급 x 만기) 매트릭스")
    st.caption("셀 = (열 계열) − (행 계열) 스프레드. 색상은 Z-score(±3 기준), 텍스트는 스프레드(bp)와 Z-score.")
    _render_heatmap(entities, z_mat, text, hover)

    # -- Sector-only matrix (rating ignored) ---------------------------------
    st.markdown("---")
    st.markdown("#### 섹터 매트릭스 (등급 무시, 섹터 내 평균)")
    st.caption("등급을 무시하고 섹터 내 모든 계열의 평균 금리로 계산한, 선택한 계열 x 만기 조합의 섹터 간 상대 스프레드입니다.")

    sec_seen = set()
    sec_rows = []
    for cat, tenor in pairs:
        if cat not in cat_meta.index:
            continue
        sector = cat_meta.loc[cat, "sector"]
        if (sector, tenor) in sec_seen:
            continue
        sec_seen.add((sector, tenor))
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

    # -- Duration Spread -------------------------------------------------
    st.markdown("---")
    st.markdown("### Duration Spread")
    st.caption("스프레드 ÷ 조정 듀레이션(6개월 이표, 액면발행 가정) — 만기별 스프레드의 '위험 대비 캐리'를 비교합니다.")

    default_base = next((c for c in all_cats if "국고채" in c), all_cats[0])
    base_cat = st.selectbox("기준(Base) 계열", all_cats,
                             index=all_cats.index(default_base) if default_base in all_cats else 0,
                             key="dur_base_cat")

    dur_pairs = category_tenor_picker(all_cats, TENOR_LABELS, "dur", default="all")
    if not dur_pairs:
        st.info("표시할 계열 x 만기 조합을 하나 이상 선택하세요.")
        return

    def _dur_entity_key(cat: str) -> tuple:
        info = cat_meta.loc[cat]
        return (sector_rank.get(info["sector"], 999), _rating_sort_key(info["rating"]))

    dur_entities = sorted({c for c, _ in dur_pairs if c != base_cat}, key=_dur_entity_key)
    dur_sel_tenors = sorted({t for _, t in dur_pairs}, key=TENOR_LABELS.index)

    if not dur_entities:
        st.warning("표시할 계열이 없습니다.")
        return

    dur_z_mat = np.full((len(dur_entities), len(dur_sel_tenors)), np.nan)
    dur_text = [["" for _ in dur_sel_tenors] for _ in dur_entities]
    dur_hover = [["" for _ in dur_sel_tenors] for _ in dur_entities]

    for i, cat in enumerate(dur_entities):
        for j, tenor in enumerate(dur_sel_tenors):
            ent_y = _latest_yield(dff, cat, tenor)
            base_y = _latest_yield(dff, base_cat, tenor)
            if np.isnan(ent_y) or np.isnan(base_y):
                dur_hover[i][j] = f"{cat} {tenor}: 데이터 없음"
                continue
            sp_bp = (ent_y - base_y) * 100
            dur = macaulay_duration_years(TENOR_YEARS[tenor], ent_y)
            if not dur or dur <= 0:
                dur_hover[i][j] = f"{cat} {tenor}: 듀레이션 계산 불가"
                continue
            ratio = sp_bp / dur
            dur_z_mat[i, j] = ratio
            dur_text[i][j] = f"{ratio:+.0f}"
            dur_hover[i][j] = (
                f"{cat} {tenor} vs {base_cat}<br>"
                f"스프레드: {sp_bp:+.1f}bp<br>듀레이션: {dur:.2f}y<br>"
                f"스프레드/듀레이션: {ratio:+.1f}bp/y"
            )

    empty_tenors = [t for j, t in enumerate(dur_sel_tenors) if np.all(np.isnan(dur_z_mat[:, j]))]
    if empty_tenors:
        st.warning(f"선택한 기간/섹터에 다음 만기 데이터가 없습니다: {', '.join(empty_tenors)}")

    st.markdown(
        f'<div style="background:#F7F8F5;border-radius:6px;padding:14px 18px;'
        f'border-left:4px solid {DEEP_GREEN};margin:12px 0">'
        f'<div style="font-size:11px;color:#888;margin-bottom:4px">종합</div>'
        f'<div style="font-size:14px;font-weight:600;color:{DEEP_GREEN}">'
        f'기준: {base_cat} &nbsp;|&nbsp; {len(dur_entities)}개 계열 x {len(dur_sel_tenors)}개 만기</div></div>',
        unsafe_allow_html=True,
    )
    st.caption("셀 = (스프레드 vs 기준, bp) ÷ (듀레이션, 년). 값이 클수록 위험(듀레이션) 대비 캐리가 두텁다는 의미입니다.")

    n_dur = len(dur_entities)
    dur_cell_text = dur_text
    if n_dur > 25:
        dur_cell_text = None
        st.caption(f"⚠ {n_dur}개 계열로 셀이 조밀해 숫자 표시는 생략했습니다 — 마우스오버로 값을 확인하거나 계열을 줄여보세요.")

    dur_heatmap_kwargs = dict(
        z=dur_z_mat.tolist(), x=dur_sel_tenors, y=dur_entities,
        hovertext=dur_hover, hoverinfo="text",
        colorscale=HEATMAP_DIVERG, zmid=0, showscale=True,
        colorbar=dict(title=dict(text="bp/y", side="right"), thickness=12, len=0.8),
    )
    if dur_cell_text is not None:
        dur_heatmap_kwargs.update(text=dur_cell_text, texttemplate="%{text}", textfont=dict(size=11))

    dur_fig = go.Figure(go.Heatmap(**dur_heatmap_kwargs))
    dur_fig.update_layout(
        height=max(320, n_dur * 40 + 90),
        margin=dict(l=150, r=30, t=10, b=10),
        font=dict(family="Apple SD Gothic Neo, Noto Sans KR, sans-serif", size=11),
        xaxis=dict(side="top"),
        plot_bgcolor="white", paper_bgcolor="white",
    )

    dur_export_cfg = copy.deepcopy(PLOTLY_CONFIG)
    dur_export_cfg["toImageButtonOptions"]["width"] = max(640, len(dur_sel_tenors) * 110)
    dur_export_cfg["toImageButtonOptions"]["height"] = max(390, n_dur * 70)
    dur_export_cfg["toImageButtonOptions"]["scale"] = 2
    st.plotly_chart(dur_fig, use_container_width=True, config=dur_export_cfg)
