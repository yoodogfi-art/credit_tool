"""Start x End forward-rate matrix (per model) and its Excel export."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd


def month_grid(horizon_m: int = 36) -> list[int]:
    """Monthly to 12m, then quarterly to the horizon: 1,2,..,12,15,18,..."""
    return [m for m in range(1, 13)] + list(range(15, horizon_m + 1, 3))


def month_label(m: int) -> str:
    return f"{m // 12}y" if m % 12 == 0 else f"{m}m"


def forward_matrix(fit, horizon_m: int = 36) -> pd.DataFrame:
    """Rows 'start Xm' (1m .. horizon-3m), cols 'end Ym' (2m .. horizon).
    Cell = forward rate (%, semi-annual comp.) from start to end; blank if end <= start."""
    grid = month_grid(horizon_m)
    starts = grid[:-1]
    ends = grid[1:]
    data = np.full((len(starts), len(ends)), np.nan)
    for i, s in enumerate(starts):
        e = np.array([x for x in ends if x > s], float)
        if len(e):
            vals = fit.forward_sa(np.full_like(e, s / 12.0), e / 12.0)
            data[i, len(ends) - len(e):] = vals
    return pd.DataFrame(data,
                        index=[f"start {month_label(s)}" for s in starts],
                        columns=[f"end {month_label(e)}" for e in ends])


def matrices_to_excel(mats: dict[str, pd.DataFrame], font_name: str = "나눔스퀘어",
                      note: str = "") -> bytes:
    """One sheet per model, laid out like the source table (A1 blank,
    'end ..' header row, 'start ..' label column, 0.000 format)."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font
    from openpyxl.utils import get_column_letter

    wb = Workbook()
    wb.remove(wb.active)
    font = Font(name=font_name, size=10)
    right = Alignment(horizontal="right")
    for name, df in mats.items():
        ws = wb.create_sheet(title=name[:31])
        ws.cell(row=1, column=1, value=None).font = font
        for j, col in enumerate(df.columns, start=2):
            c = ws.cell(row=1, column=j, value=col)
            c.font = font
        for i, (idx, row) in enumerate(df.iterrows(), start=2):
            c = ws.cell(row=i, column=1, value=idx)
            c.font = font
            for j, v in enumerate(row.tolist(), start=2):
                if pd.isna(v):
                    continue
                c = ws.cell(row=i, column=j, value=round(float(v), 6))
                c.font = font
                c.number_format = "0.000"
                c.alignment = right
        ws.column_dimensions["A"].width = 10
        for j in range(2, len(df.columns) + 2):
            ws.column_dimensions[get_column_letter(j)].width = 8.5
        ws.freeze_panes = "B2"
        if note:
            c = ws.cell(row=len(df) + 3, column=1, value=note)
            c.font = Font(name=font_name, size=9, color="808080")
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()
