"""Page: Forward Rate — spot & forward curves from a bond-level treasury
snapshot, fitted with four models (Nelson-Siegel, Svensson, Cubic Spline,
Bootstrapping)."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from assets.styles import CORAL, DEEP_GREEN, GS_GOLD, OLIVE
from chart_utils import base_layout, show_chart
from data.treasury_loader import flag_on_the_run, infer_valuation_date, with_source
from models.forward_matrix import forward_matrix, matrices_to_excel
from models.term_structure import MODEL_NAMES, fit_all, robust_weights

MODEL_COLORS = {
    "Nelson-Siegel": DEEP_GREEN,
    "Svensson":      GS_GOLD,
    "Cubic Spline":  "#6B7B9A",
    "Bootstrapping": CORAL,
}
MODEL_DASH = {"Nelson-Siegel": "solid", "Svensson": "solid", "Cubic Spline": "dash", "Bootstrapping": "dot"}

# (start, length) in years -> label "1Y×1Y" etc.
STD_FORWARDS = [(0.25, 0.25), (0.5, 0.5), (1, 1), (2, 1), (3, 1), (4, 1), (1, 2), (2, 3),
                (3, 2), (5, 5), (10, 10), (20, 10)]
SPOT_TENORS = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30, 50]


def _ylab(t: float) -> str:
    if t < 1:
        return f"{int(round(t * 12))}M"
    return f"{t:g}Y"


def _fwd_label(s: float, L: float) -> str:
    return f"{_ylab(s)}×{_ylab(L)}"


def _fmt_df(df: pd.DataFrame, fmt: str = "{:.3f}") -> pd.DataFrame:
    return df.map(lambda v: "-" if pd.isna(v) else fmt.format(v))


def _to_semiannual(y_pct: pd.Series, freq: pd.Series) -> pd.Series:
    """Re-express a YTM quoted with compounding frequency m (1, 4, 12) as the
    equivalent semi-annual rate, so bonds with different coupon frequencies sit
    on one curve. m = 2, 0 (pay at maturity) or missing is left unchanged."""
    m = freq.where(freq.isin([1, 4, 12]))
    cc = m * np.log1p(y_pct / (100 * m))
    conv = 200 * np.expm1(cc / 2)
    return conv.where(m.notna(), y_pct)


def render(bonds_all: pd.DataFrame, sources: list[str], trade_date: pd.Timestamp | None = None) -> None:
    st.header("Forward Rate")

    # ---------------- pricing source (one source per curve; never mixed) ----------------
    src_n = {s: int(bonds_all[f"ytm_{s}"].notna().sum()) for s in sources}
    src = st.radio(
        "금리 소스", sources, horizontal=True, key="fw_src",
        format_func=lambda s: f"{s} ({src_n[s]:,}종목)",
        help="Bloomberg 가격 소스별 YTM·듀레이션. 한 커브에는 한 소스만 씁니다(소스 간 수 bp 차이가 노이즈가 되므로 섞지 않음).",
    ) if len(sources) > 1 else sources[0]
    bonds = with_source(bonds_all, src)

    # ---------------- ticker selection (nothing is fitted until the user picks) ----------------
    summ = (bonds.groupby("ticker")
            .agg(name=("name", lambda x: x.mode().iat[0] if len(x.mode()) else ""),
                 n=("ytm", "size"), mat_min=("maturity", "min"), mat_max=("maturity", "max"),
                 ytm=("ytm", "median"))
            .sort_values("n", ascending=False))
    # order: KTB first, then other K* tickers, then A, B, C, ... (alphabetical within each letter)
    def _tk_order(tk: str) -> tuple:
        u = tk.upper()
        return (0 if u == "KTB" else 1 if u.startswith("K") else 2, u)

    summ = summ.loc[sorted(summ.index, key=_tk_order)]
    tickers = summ.index.tolist()

    def _tk_label(tk: str) -> str:
        r = summ.loc[tk]
        nm = f" · {r['name']}" if r["name"] and r["name"] != tk else ""
        return f"{tk}{nm} ({r['n']}종목)"

    with st.expander(f"파일 내 발행자 {len(tickers)}개 ({src} 금리 있는 종목 기준)", expanded=False):
        st.dataframe(
            pd.DataFrame({
                "발행자명": summ["name"], "종목 수": summ["n"],
                "최단 만기": summ["mat_min"].dt.strftime("%Y-%m-%d"),
                "최장 만기": summ["mat_max"].dt.strftime("%Y-%m-%d"),
                "YTM 중앙값(%)": summ["ytm"].round(3),
            }),
            width="stretch",
        )

    c1, c2, c3 = st.columns([3, 2, 2])
    sel_tickers = c1.multiselect(
        "발행자(Ticker) — 여러 개 선택 시 하나의 커브로 합쳐서 적합", tickers,
        default=[], format_func=_tk_label, placeholder="커브를 그릴 발행자를 선택하세요",
        key="fw_tickers",
    )
    if not sel_tickers:
        st.info("발행자를 선택하면 커브와 선도금리를 계산합니다.")
        return
    ticker = "+".join(sel_tickers) if len(sel_tickers) <= 2 else f"{sel_tickers[0]} 외 {len(sel_tickers) - 1}개"
    sub = bonds[bonds["ticker"].isin(sel_tickers)].copy()

    # valuation date: the date the source's analytics are computed from, backed out of the
    # source's KTB durations (plain semi-annual bullets — other issuers' conventions vary),
    # else the trade date in the file name
    ktb = bonds[bonds["ticker"] == "KTB"]
    vd_dur = infer_valuation_date(ktb if len(ktb) else sub)
    vd_default = vd_dur or trade_date or pd.Timestamp.today().normalize()
    val_date = pd.Timestamp(c2.date_input("기준일", value=vd_default.date(), key=f"fw_vd_{src}_{ticker}"))
    sel_models = c3.multiselect("모델", MODEL_NAMES, default=MODEL_NAMES, key="fw_models")
    notes = []
    if trade_date is not None:
        notes.append(f"파일명 거래일 {trade_date:%Y-%m-%d}")
    if vd_dur is not None:
        notes.append(f"{src} 듀레이션 역산 {vd_dur:%Y-%m-%d}")
    if notes:
        c2.caption(" · ".join(notes))

    sub["t"] = (sub["maturity"] - val_date).dt.days / 365.0
    sub = sub[sub["t"] > 1 / 365].sort_values("t")
    if len(sub) < 6:
        st.warning(f"{ticker}: 유효한 채권이 {len(sub)}개뿐입니다 (최소 6개 필요).")
        return
    sub["ytm_raw"] = sub["ytm"]
    sub["ytm"] = _to_semiannual(sub["ytm"], sub["freq"])
    n_conv = int((sub["ytm"] - sub["ytm_raw"]).abs().gt(1e-9).sum())
    sub["otr"] = flag_on_the_run(sub)
    if n_conv:
        st.caption(f"이표 주기가 연 1·4·12회인 {n_conv}종목은 YTM을 반기복리 등가로 환산해 적합합니다.")

    t_max_data = float(sub["t"].max())
    o1, o2, o3, o4, o5 = st.columns([2, 2, 2, 2, 1])
    min_t = o1.select_slider("최단 만기 제외 (년 미만)", options=[0.0, 0.1, 0.25, 0.5, 1.0], value=0.0, key="fw_min_t")
    h_opts = [x for x in (5, 10, 20, 30, 50) if x < t_max_data + 5] or [5]
    horizon = o2.select_slider("차트 표시 만기 (년)", options=h_opts,
                               value=30 if 30 in h_opts else h_opts[-1], key="fw_horizon")
    has_otr = bool(sub["otr"].any())
    otr_mode = o3.radio(
        "지표물", ["포함", "제외"], index=0, horizontal=True, key="fw_otr", disabled=not has_otr,
        help="만기별(2·3·5·10·20·30·50년) 가장 최근 발행 종목. 발행일(issue_dt)로 판정합니다. "
             "유동성 프리미엄을 커브에서 빼고 싶으면 '제외'.")
    out_mode = o4.radio(
        "이상치 처리", ["가중치 축소", "제외", "없음"], index=0, horizontal=True, key="fw_out_mode",
        help="Svensson 기준 잔차가 임계값을 넘는 종목을 가중치 축소(Huber) 또는 제외한 뒤 4개 모델을 다시 적합합니다.")
    out_thr = o5.number_input("임계값(bp)", value=5.0, min_value=1.0, max_value=50.0, step=1.0,
                              key="fw_out_thr", disabled=out_mode == "없음")

    fit_df = sub[sub["t"] >= min_t] if min_t > 0 else sub
    otr_excl = fit_df[fit_df["otr"]] if (has_otr and otr_mode == "제외") else fit_df.iloc[0:0]
    fit_df = fit_df.drop(otr_excl.index)
    if len(fit_df) < 6:
        st.warning("제외 조건 적용 후 채권 수가 너무 적습니다.")
        return
    if not sel_models:
        st.info("모델을 하나 이상 선택하세요.")
        return

    t_obs = fit_df["t"].to_numpy()
    y_obs = fit_df["ytm"].to_numpy()
    mode_key = {"가중치 축소": "downweight", "제외": "exclude", "없음": "none"}[out_mode]
    w_obs, res_ref = robust_weights(t_obs, y_obs, mode_key, float(out_thr))
    fits = fit_all(t_obs, y_obs, None if mode_key == "none" else w_obs)
    fits = {m: fits[m] for m in sel_models}
    flagged = fit_df.assign(w=w_obs, res=res_ref)[w_obs < 0.999]

    # ---------------- metrics ----------------
    w_rm = np.maximum(w_obs, 1e-9)   # RMSE over the bonds the fit actually used
    rmse = {m: float(np.sqrt(np.average((f.fitted_obs - y_obs) ** 2, weights=w_rm)) * 100)
            for m, f in fits.items()}
    mcols = st.columns(len(fits) + 1)
    mcols[0].metric("적합 채권 수", f"{len(fit_df)}개",
                    help=f"{src} · 기준일 {val_date:%Y-%m-%d}, 최장 {t_max_data:.1f}년"
                         + (f" · 지표물 {len(otr_excl)}개 제외" if len(otr_excl) else ""))
    for col, (m, v) in zip(mcols[1:], rmse.items()):
        col.metric(f"{m} RMSE", f"{v:.1f}bp")

    grid = np.linspace(max(0.05, min(t_obs.min(), 0.25)), horizon, 400)
    date_sub = f"{src} · 기준일 {val_date:%Y-%m-%d} · {len(fit_df)}개 종목"

    # ---------------- chart 1: spot curve ----------------
    fig = go.Figure()
    in_rng = fit_df.assign(w=w_obs)[fit_df["t"] <= horizon]
    normal = in_rng[(in_rng["w"] >= 0.999) & ~in_rng["otr"]]
    flag_pts = in_rng[(in_rng["w"] < 0.999) & ~in_rng["otr"]]
    otr_pts = pd.concat([in_rng[in_rng["otr"]], otr_excl[otr_excl["t"] <= horizon].assign(w=0.0)])
    fig.add_trace(go.Scatter(
        x=normal["t"], y=normal["ytm"], mode="markers", name="관측 YTM",
        marker=dict(size=5, color="#9E9E9E", line=dict(width=0)),
        text=normal["desc"], hovertemplate="%{text}<br>%{x:.2f}y: %{y:.3f}%<extra></extra>",
    ))
    if len(flag_pts):
        fig.add_trace(go.Scatter(
            x=flag_pts["t"], y=flag_pts["ytm"], mode="markers",
            name="이상치 (제외)" if mode_key == "exclude" else "이상치 (가중치 축소)",
            marker=dict(size=8, color="rgba(0,0,0,0)", line=dict(width=1.5, color=CORAL), symbol="circle"),
            text=flag_pts["desc"], customdata=flag_pts["w"],
            hovertemplate="%{text}<br>%{x:.2f}y: %{y:.3f}%<br>가중치 %{customdata:.2f}<extra></extra>",
        ))
    if len(otr_pts):
        fig.add_trace(go.Scatter(
            x=otr_pts["t"], y=otr_pts["ytm"], mode="markers",
            name="지표물 (제외)" if otr_mode == "제외" else "지표물",
            marker=dict(size=10, symbol="diamond" if otr_mode == "포함" else "diamond-open",
                        color=DEEP_GREEN, line=dict(width=1.5, color=DEEP_GREEN)),
            text=otr_pts["desc"], customdata=otr_pts["w"],
            hovertemplate="%{text} (지표물)<br>%{x:.2f}y: %{y:.3f}%<br>가중치 %{customdata:.2f}<extra></extra>",
        ))
    for m, f in fits.items():
        fig.add_trace(go.Scatter(
            x=grid, y=f.spot_sa(grid), name=m, mode="lines",
            line=dict(color=MODEL_COLORS[m], width=2, dash=MODEL_DASH[m]),
            hovertemplate=f"{m}: %{{y:.3f}}%<extra></extra>",
        ))
    base_layout(fig, "", 420)
    fig.update_layout(hovermode="x unified")
    fig.update_xaxes(title_text="만기(년)", ticksuffix="y", showgrid=True, gridcolor="#F0F0F0")
    fig.update_yaxes(title_text="금리(%)", ticksuffix="%")
    show_chart(fig, f"{ticker} 현물(Spot) 금리 커브 — 모델 {len(fits)}종", subtitle=date_sub, key="fw_spot")

    if len(flagged):
        verb = "제외" if mode_key == "exclude" else "가중치 축소"
        with st.expander(f"이상치 {len(flagged)}종목 {verb} (임계값 ±{out_thr:g}bp, Svensson 잔차 기준)", expanded=False):
            st.dataframe(pd.DataFrame({
                "종목": flagged["desc"], "만기": flagged["maturity"].dt.strftime("%Y-%m-%d"),
                "YTM(%)": flagged["ytm"].round(3), "잔차(bp)": flagged["res"].round(1),
                "가중치": flagged["w"].round(2),
            }).set_index("종목"), width="stretch")
            st.caption("잔차 < 0 = 커브보다 금리가 낮음(비쌈) — 주로 지표물의 유동성 프리미엄.")

    # ---------------- chart 2: forward curves (one chart per selected term) ----------------
    fwd_terms = {"순간": 0.0, "1M": 1 / 12, "3M": 0.25, "6M": 0.5, "1Y": 1.0, "2Y": 2.0,
                 "3Y": 3.0, "5Y": 5.0, "10Y": 10.0, "20Y": 20.0}
    fwd_terms = {k: v for k, v in fwd_terms.items() if v < t_max_data - 0.25}
    sel_terms = st.pills(
        "선도금리 기간 — t년 후 시작하는 해당 기간 금리 (여러 개 선택 시 차트가 각각 그려집니다)",
        list(fwd_terms), selection_mode="multi", default=["1Y"] if "1Y" in fwd_terms else list(fwd_terms)[:1],
        key="fw_terms",
        help="순간 = instantaneous forward f(t). 예) 1Y 선택 시 x축 2y 지점 = 2년 후 시작하는 1년물(2Y×1Y).",
    )
    ref_m = next((m for m in ("Svensson", "Nelson-Siegel", "Cubic Spline") if m in fits), None)
    for term in sorted(sel_terms or [], key=lambda k: fwd_terms[k]):
        L = fwd_terms[term]
        fg = grid if L == 0 else grid[grid + L <= t_max_data + 1e-9]
        fig2 = go.Figure()
        for m, f in fits.items():
            y = f.inst_forward_sa(fg) if L == 0 else f.forward_sa(fg, fg + L)
            fig2.add_trace(go.Scatter(
                x=fg, y=y, name=m, mode="lines",
                line=dict(color=MODEL_COLORS[m], width=2, dash=MODEL_DASH[m],
                          shape="linear"),
                hovertemplate=f"{m}: %{{y:.3f}}%<extra></extra>",
            ))
        if ref_m:  # reference spot curve, thin gray
            fig2.add_trace(go.Scatter(
                x=grid, y=fits[ref_m].spot_sa(grid), name=f"Spot ({ref_m})", mode="lines",
                line=dict(color="#BDBDBD", width=1.5), hovertemplate="Spot: %{y:.3f}%<extra></extra>",
            ))
        base_layout(fig2, "", 420)
        fig2.update_xaxes(title_text="시작 시점 t (년)", ticksuffix="y", showgrid=True, gridcolor="#F0F0F0",
                          range=[0, horizon])
        fig2.update_yaxes(title_text="금리(%)", ticksuffix="%")
        title = (f"{ticker} 순간 선도금리 커브 — 모델 {len(fits)}종" if L == 0 else
                 f"{ticker} {term} 선도금리 커브 (t년 후 {term}물) — 모델 {len(fits)}종")
        show_chart(fig2, title, subtitle=date_sub, key=f"fw_fwd_{term}")
    
    # ---------------- table + chart 3: standard forwards ----------------
    fwd_rows = [(s, l) for s, l in STD_FORWARDS if s + l <= t_max_data + 1e-9]
    fwd_tbl = pd.DataFrame(
        {m: [float(f.forward_sa(s, s + l)[0]) for s, l in fwd_rows] for m, f in fits.items()},
        index=[_fwd_label(s, l) for s, l in fwd_rows],
    )
    if len(fits) > 1:
        fwd_tbl["모델 평균"] = fwd_tbl.mean(axis=1)
        fwd_tbl["모델 간 차이(bp)"] = (fwd_tbl[list(fits)].max(axis=1) - fwd_tbl[list(fits)].min(axis=1)) * 100

    st.markdown("---")
    t1, t2 = st.columns([3, 2])
    with t1:
        fig3 = go.Figure()
        for m in fits:
            fig3.add_trace(go.Bar(
                x=fwd_tbl.index, y=fwd_tbl[m], name=m, marker_color=MODEL_COLORS[m],
                marker_line_width=0, hovertemplate=f"{m} %{{x}}: %{{y:.3f}}%<extra></extra>",
            ))
        base_layout(fig3, "", 380)
        lo = np.nanmin(fwd_tbl[list(fits)].to_numpy()); hi = np.nanmax(fwd_tbl[list(fits)].to_numpy())
        fig3.update_layout(barmode="group", bargap=0.25, bargroupgap=0.05, hovermode="x unified")
        fig3.update_yaxes(ticksuffix="%", range=[max(0, lo - 0.3), hi + 0.15])
        fig3.update_xaxes(tickangle=0, tickfont=dict(size=10))
        show_chart(fig3, f"{ticker} 주요 선도금리 모델별 비교", subtitle=date_sub, key="fw_bar")
    with t2:
        st.markdown(f'<div style="font-weight:700;color:{DEEP_GREEN};font-size:14px;margin:6px 0 4px">'
                    f'주요 선도금리 (%, 반기복리)</div>', unsafe_allow_html=True)
        disp = _fmt_df(fwd_tbl.drop(columns=["모델 간 차이(bp)"], errors="ignore"))
        if "모델 간 차이(bp)" in fwd_tbl:
            disp["차이(bp)"] = fwd_tbl["모델 간 차이(bp)"].map(lambda v: f"{v:.1f}")
        st.dataframe(disp, width="stretch", height=min(460, 38 + 35 * len(disp)))
        st.caption("A×B = A 시점부터 B 기간의 선도금리. 예) 1Y×1Y = 1년 후 1년물.")

    # ---------------- start x end forward matrix (per model) ----------------
    st.markdown("---")
    h1, h2 = st.columns([3, 1])
    h1.markdown(f'<div style="font-weight:700;color:{DEEP_GREEN};font-size:14px;margin:6px 0 0">'
                f'선도금리 매트릭스 (start × end, %, 반기복리)</div>', unsafe_allow_html=True)
    h_opts_m = [m for m in (36, 60, 120) if m / 12 <= t_max_data]
    horizon_m = h2.selectbox("최장 end", h_opts_m or [36], format_func=lambda m: f"{m // 12}y",
                             key="fw_mat_h")
    mats = {m: forward_matrix(f, horizon_m) for m, f in fits.items()}
    if len(mats) > 1:
        mats["모델 평균"] = sum(mats[m] for m in fits) / len(fits)
    xlsx = matrices_to_excel(
        mats, note=f"{ticker} 기준일 {val_date:%Y-%m-%d} · {len(fit_df)}개 종목 · 반기복리(%)")
    h2.download_button("Excel 다운로드 (모델별 시트)", xlsx,
                       file_name=f"{ticker}_선도금리_매트릭스_{val_date:%Y%m%d}.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       key="fw_mat_dl", width="stretch")
    for tab, (m, mat) in zip(st.tabs(list(mats)), mats.items()):
        with tab:
            st.dataframe(_fmt_df(mat).replace("-", ""), width="stretch",
                         height=38 + 35 * len(mat))

    # ---------------- spot / instantaneous forward by tenor ----------------
    st.markdown("---")
    tens = [x for x in SPOT_TENORS if x <= t_max_data + 1e-9]
    spot_tbl = pd.DataFrame({m: f.spot_sa(np.array(tens)) for m, f in fits.items()},
                            index=[_ylab(x) for x in tens]).T
    st.markdown(f'<div style="font-weight:700;color:{DEEP_GREEN};font-size:14px;margin:6px 0 4px">'
                f'만기별 Spot 금리 (%, 반기복리)</div>', unsafe_allow_html=True)
    st.dataframe(_fmt_df(spot_tbl), width="stretch")

    # ---------------- residuals ----------------
    with st.expander("모델 적합 잔차 / 파라미터", expanded=False):
        fig4 = go.Figure()
        for m, f in fits.items():
            fig4.add_trace(go.Scatter(
                x=t_obs, y=(y_obs - f.fitted_obs) * 100, mode="markers", name=m,
                marker=dict(size=5, color=MODEL_COLORS[m]),
                text=fit_df["desc"], hovertemplate=f"{m}<br>%{{text}}: %{{y:+.1f}}bp<extra></extra>",
            ))
        base_layout(fig4, "", 340)
        fig4.update_layout(hovermode="closest")
        fig4.update_xaxes(title_text="만기(년)", ticksuffix="y")
        fig4.update_yaxes(title_text="관측 − 모델(bp)", ticksuffix="bp", zeroline=True, zerolinecolor="#AAAAAA")
        show_chart(fig4, f"{ticker} 모델 적합 잔차", subtitle=date_sub, key="fw_resid")

        prm = []
        for m, f in fits.items():
            p = {k: (round(float(v), 4) if isinstance(v, (int, float, np.floating)) else v)
                 for k, v in f.params.items()}
            prm.append({"모델": m, "RMSE(bp)": round(rmse[m], 2),
                        "파라미터": ", ".join(f"{k}={v}" for k, v in p.items())})
        st.dataframe(pd.DataFrame(prm).set_index("모델"), width="stretch")
        st.caption(
            "NS·Svensson·Cubic Spline은 YTM(반기복리)을 연속복리로 변환해 직접 적합(YTM≈Spot 근사), "
            "Bootstrapping은 지표 만기(3M·6M·1Y·2Y·3Y·5Y·7Y·10Y·15Y·20Y·30Y·50Y)별 par 금리를 인근 종목으로 추정하고, "
            "6개월 이표 par 채권으로 구간별 선도금리를 bootstrap한 뒤 Hagan-West monotone convex 보간으로 "
            "연속적인 순간 선도금리 커브를 만듭니다(구간 평균 보존 → 할인계수 유지, 노드에서 끊김 없음). "
            "선도금리는 모두 연속복리 Spot에서 계산 후 반기복리로 표시합니다."
        )
