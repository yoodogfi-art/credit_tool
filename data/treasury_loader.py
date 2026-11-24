"""
Bond-level snapshot loader (Bloomberg BQL exports).

Supported layouts
  * Workbook with a 'final' sheet (BQL `get(...)` export: row 1 = query,
    row 2 = headers like `yield(yt='ytm',pricing_source='LCPR',side='mid').value`)
    plus optionally 'Sheet1' (older BQL export, headers like
    `YIELD(YT=YTM,PRICING_SOURCE=BVAL,SIDE=MID)`). Rows are joined on the
    security ID (first column, e.g. 'AF122058 Corp').
  * Single-sheet workbook (first sheet) in either header style.

Each pricing source found in the headers (BVAL, LCPR, ...) keeps its own
yield / duration columns (ytm_<SRC>, mac_<SRC>, mod_<SRC>) — sources are never
mixed row by row; the page picks one source per curve. Static fields
(issue date, amount outstanding, coupon type/frequency, issuer) come from
the 'final' sheet when present.
"""

from __future__ import annotations

import io
import re

import numpy as np
import pandas as pd
import streamlit as st

NA_VALUES = {"#N/A": np.nan, "#N/A N/A": np.nan, "#NAME?": np.nan, "#N/A Field Not Applicable": np.nan}
EXCLUDE_CPN_TYP = {"FLOATING", "VARIABLE", "PRELIM", "STEP CPN", "FLAT TRADING", "PAY-IN-KIND"}


# ---------------------------------------------------------------------------
# Header normalisation
# ---------------------------------------------------------------------------
def _norm(h) -> str:
    s = str(h).strip().lower()
    s = s.replace("().value", "").replace(".value", "").replace("'", "").replace('"', "").replace(" ", "")
    return s


def _classify(h) -> tuple[str | None, str | None]:
    """Return (field key, pricing source) for a raw header."""
    n = _norm(h)
    base = n.split("(")[0]
    m = re.search(r"pricing_source=([a-z0-9_]+)", n)
    src = m.group(1).upper() if m else None
    if base in ("id", "security", "id_bb_global"):
        return "id", None
    if base == "security_des":
        return "desc", None
    if base == "long_comp_name":
        return "name", None
    if base == "issuer_bulk":
        return "issuer", None
    if base == "ticker":
        return "ticker", None
    if base == "cpn":
        return "cpn", None
    if base == "maturity":
        return "maturity", None
    if base == "amt_outstanding":
        return "amt", None
    if base == "issue_dt":
        return "issue_dt", None
    if base == "cpn_typ":
        return "cpn_typ", None
    if base == "cpn_freq":
        return "freq", None
    if base == "crncy":
        return "crncy", None
    if base == "duration":
        if "macaulay" in n:
            return "mac", src
        if "modified" in n:
            return "mod", src
    if base == "yield":
        return "ytm", src
    if base in ("price", "px_dirty_mid") and ("dirty" in n or base == "px_dirty_mid"):
        return "dirty", src
    return None, None


def _parse_sheet(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Parse one sheet into normalised columns. Source-specific fields are
    suffixed (_BVAL, _LCPR, ...). Returns (frame, sources found)."""
    hdr_row = None
    for i in range(min(10, len(raw))):
        cells = [str(v) for v in raw.iloc[i].tolist() if pd.notna(v)]
        low = " ".join(c.lower() for c in cells)
        if len(cells) >= 4 and "maturity" in low and ("yield" in low or "ytm" in low):
            hdr_row = i
            break
    if hdr_row is None:
        return pd.DataFrame(), []

    headers = raw.iloc[hdr_row].tolist()
    body = raw.iloc[hdr_row + 1:].reset_index(drop=True)
    out, sources = {}, []
    for j, h in enumerate(headers):
        key, src = _classify(h) if pd.notna(h) else (None, None)
        if j == 0 and key is None:
            key = "id"                                  # BQL puts the security ID first
        if key is None:
            continue
        if key in ("ytm", "mac", "mod", "dirty"):
            src = src or "DEFAULT"
            if src not in sources:
                sources.append(src)
            key = f"{key}_{src}"
        if key not in out:
            out[key] = body.iloc[:, j]
    df = pd.DataFrame(out).replace(NA_VALUES)
    return df, sources


def trade_date_from_filename(name: str | None) -> pd.Timestamp | None:
    if not name:
        return None
    m = re.search(r"(20\d{2})[-_]?(\d{2})[-_]?(\d{2})", name)
    if not m:
        return None
    try:
        return pd.Timestamp(f"{m.group(1)}-{m.group(2)}-{m.group(3)}")
    except ValueError:
        return None


# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_treasury(file_bytes: bytes) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Return (bonds, sources, warnings).

    bonds: id, desc, name, ticker, cpn, maturity, amt, issue_dt, cpn_typ, freq,
           crncy, plus ytm_<SRC>, mac_<SRC>, mod_<SRC>, dirty_<SRC> per source.
    sources: pricing sources found, preferred first (BVAL before others).
    """
    xls = pd.ExcelFile(io.BytesIO(file_bytes))
    names = xls.sheet_names
    order = (["final"] if "final" in names else []) + [n for n in names if n != "final"]
    if "final" not in names:
        order = names[:1]

    frames, sources = [], []
    for sh in order:
        raw = pd.read_excel(xls, sheet_name=sh, header=None)
        df, srcs = _parse_sheet(raw)
        if df.empty or "id" not in df:
            continue
        frames.append((sh, df))
        sources += [s for s in srcs if s not in sources]
    if not frames:
        raise ValueError("헤더(maturity, yield 열)를 찾을 수 없습니다.")

    # join on security ID; static fields from the first (final) sheet win
    base = frames[0][1].drop_duplicates("id").set_index("id")
    for _, df in frames[1:]:
        df = df.drop_duplicates("id").set_index("id")
        new_cols = [c for c in df.columns if c not in base.columns]
        base = base.join(df[new_cols], how="outer")
        for c in df.columns:                      # fill static gaps only
            if c in base.columns and c not in new_cols:
                base[c] = base[c].combine_first(df[c])
    bonds = base.reset_index()

    for c in ("desc", "name", "issuer", "ticker", "cpn_typ", "crncy"):
        if c not in bonds:
            bonds[c] = ""
        bonds[c] = bonds[c].astype(str).str.strip().replace({"nan": "", "None": ""})
    if (bonds["name"] == "").all():
        bonds["name"] = bonds["issuer"]
    for c in ("cpn", "amt", "freq"):
        bonds[c] = pd.to_numeric(bonds[c], errors="coerce") if c in bonds else np.nan
    for c in ("maturity", "issue_dt"):
        bonds[c] = pd.to_datetime(bonds[c], errors="coerce") if c in bonds else pd.NaT
    for src in sources:
        for k in ("ytm", "mac", "mod", "dirty"):
            col = f"{k}_{src}"
            bonds[col] = pd.to_numeric(bonds[col], errors="coerce") if col in bonds else np.nan

    warnings: list[str] = []
    n0 = len(bonds)
    bonds = bonds.dropna(subset=["maturity"])
    bad_typ = bonds["cpn_typ"].str.upper().isin(EXCLUDE_CPN_TYP)
    is_float = bonds["desc"].str.contains(r"\bFloat\b|\bFRN\b", case=False, regex=True)
    bonds = bonds[~bad_typ & ~is_float]
    if n0 - len(bonds):
        warnings.append(f"__toast__만기 없음·변동/스텝/예비 종목 {n0 - len(bonds)}건 제외")
    # KRW bonds only (the tool never analyses other currencies); applied whenever
    # the file carries a currency column
    if (bonds["crncy"] != "").any():
        n1 = len(bonds)
        bonds = bonds[bonds["crncy"].str.upper() == "KRW"]
        if n1 - len(bonds):
            warnings.append(f"__toast__원화 외 통화 {n1 - len(bonds)}건 제외")

    sources = sorted(sources, key=lambda s: (s != "BVAL", s))
    if not any(bonds[f"ytm_{s}"].notna().any() for s in sources):
        raise ValueError("금리(yield)가 있는 종목이 없습니다.")
    return bonds.reset_index(drop=True), sources, warnings


def with_source(bonds: pd.DataFrame, src: str) -> pd.DataFrame:
    """View of bonds priced by one source: adds ytm / mac_dur / mod_dur / dirty
    columns from that source and drops rows without a yield."""
    out = bonds.copy()
    out["ytm"] = out[f"ytm_{src}"]
    out["mac_dur"] = out.get(f"mac_{src}", np.nan)
    out["mod_dur"] = out.get(f"mod_{src}", np.nan)
    out["dirty"] = out.get(f"dirty_{src}", np.nan)
    return out.dropna(subset=["ytm"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# On-the-run (지표물) — most recent new issue per original tenor
# ---------------------------------------------------------------------------
KTB_TENORS = [2, 3, 5, 10, 20, 30, 50]


def flag_on_the_run(bonds: pd.DataFrame, tenors: list[int] = KTB_TENORS) -> pd.Series:
    """True for the latest-issued bond in each original-tenor bucket.
    Original tenor = maturity - issue date, snapped to the nearest standard
    tenor within ±max(0.5y, 10%)."""
    flag = pd.Series(False, index=bonds.index)
    if "issue_dt" not in bonds or bonds["issue_dt"].isna().all():
        return flag
    orig = (bonds["maturity"] - bonds["issue_dt"]).dt.days / 365.25
    bucket = pd.Series(np.nan, index=bonds.index)
    for T in tenors:
        m = (orig - T).abs() <= max(0.5, 0.1 * T)
        bucket[m & bucket.isna()] = T
    for T, grp in bonds[bucket.notna()].groupby(bucket[bucket.notna()]):
        flag[grp["issue_dt"].idxmax()] = True
    return flag


def infer_valuation_date(bonds: pd.DataFrame) -> pd.Timestamp | None:
    """Estimate the analytics date from bullet bonds in their final coupon
    period, where Macaulay duration (years) == time to maturity."""
    s = bonds.dropna(subset=["mac_dur"])
    s = s[(s["mac_dur"] > 0) & (s["mac_dur"] < 0.45)]
    if "freq" in s:                                   # skip amortisers / odd payers
        s = s[s["freq"].isin([1, 2]) | s["freq"].isna()]
    if s.empty:
        return None
    est = s["maturity"] - pd.to_timedelta(s["mac_dur"] * 365.0, unit="D")
    d = pd.Timestamp(est.median()).round("D")
    if d.weekday() == 5:
        d -= pd.Timedelta(days=1)
    elif d.weekday() == 6:
        d += pd.Timedelta(days=1)
    return d
