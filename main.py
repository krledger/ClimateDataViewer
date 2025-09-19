#!/usr/bin/env python3
# metrics_from_parquet.py — read raw_daily.parquet in FOLDER → metrics.parquet + metrics.csv
# Canonical label: Type (Name, Location).  Normalises "5 Day" / "5-Day" to "5-Day".

import os, sys, json, traceback, warnings, re
from datetime import datetime
import numpy as np
import pandas as pd

# ====== EDIT THIS (or set env CLIMATE_FOLDER) ======
FOLDER = "/Users/ken/Documents/Climate Analysis/SSP5-85"
# ===================================================

# Metric settings
WET_DAY_MM = 1.0
R95_BASELINE_START, R95_BASELINE_END = 2015, 2099

warnings.filterwarnings("ignore", category=FutureWarning)

def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

# ---------- season helpers ----------
def add_season_fields(idx: pd.Index) -> pd.DataFrame:
    dti = pd.to_datetime(idx)
    m = dti.month.values
    season = np.select(
        [np.isin(m, [12,1,2]), np.isin(m, [3,4,5]), np.isin(m, [6,7,8]), np.isin(m, [9,10,11])],
        ["DJF", "MAM", "JJA", "SON"],
        default="UNK",
    )
    season = pd.Categorical(season, categories=["DJF", "MAM", "JJA", "SON", "UNK"], ordered=True)
    season_year = dti.year + (m == 12)
    out = pd.DataFrame({"Year": dti.year, "Season": season, "Season_Year": season_year}, index=idx)
    vc = pd.Series(season).value_counts(sort=False)
    log("Season rows → " + ", ".join(f"{k}:{int(v)}" for k, v in vc.items()))
    return out

def max_consecutive(bools: pd.Series) -> int:
    best = run = 0
    for v in bools.astype(bool).to_numpy():
        run = run + 1 if v else 0
        if run > best: best = run
    return int(best)

def pct(series: pd.Series, q: float) -> float:
    v = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
    return float(np.percentile(v, q*100)) if v.size else np.nan

# ---------- label helpers (strict canonical form) ----------
# Canonical pattern: Type (Name, Location)
_CANON_RX = re.compile(r"^(Temp|Wind|Rain) \([^)]+, [^)]+\)$")

def _norm_5day(text: str) -> str:
    # "5 Day", "5- Day", "5 -Day" → "5-Day"
    return re.sub(r"\b5\s*-\s*Day\b|\b5\s+Day\b", "5-Day", text, flags=re.I)

def canonical(name: str, typ: str, region: str) -> str:
    """Return 'Type (Name, Location)'.  Normalises whitespace and '5-Day'."""
    name = _norm_5day(re.sub(r"\s+", " ", str(name)).strip())
    region = re.sub(r"\s+", " ", str(region)).strip()
    t = str(typ).lower()
    typ_norm = {"temp": "Temp", "wind": "Wind", "rain": "Rain"}.get(t, t.title())
    return f"{typ_norm} ({name}, {region})"

def standardise_label(lbl: str) -> str:
    """Coerce any legacy label into 'Type (Name, Location)'.  Idempotent."""
    if not isinstance(lbl, str):
        return lbl

    s = _norm_5day(lbl.strip())
    s = re.sub(r"\s+", " ", s)

    # Ensure there is a space before '(' then tidy internals
    s = re.sub(r"(?<!\s)\(", " (", s)          # 'X(…' → 'X (…'
    s = re.sub(r"\(\s*", "(", s)               # '( A' → '(A'
    s = re.sub(r"\s*,\s*", ", ", s)            # ',A' or ',  A' → ', A'
    s = re.sub(r"\s*\)\s*$", ")", s)           # ' ) ' → ')'

    # Already canonical → Type (Name, Location)
    if _CANON_RX.match(s):
        return s

    # Old order: Name (Type, Location) → Type (Name, Location)
    m = re.match(r"^(?P<name>[^()]+?) \((?P<typ>Temp|Wind|Rain), (?P<loc>[^)]+)\)$", s, flags=re.I)
    if m:
        return canonical(m["name"], m["typ"], m["loc"])

    # Averages
    m = re.match(r"Average (Temp|Temperature) \((?P<loc>[^)]+)\)$", s, flags=re.I)
    if m:
        return canonical("Average", "Temp", m["loc"])
    m = re.match(r"Average Wind \((?P<loc>[^)]+)\)$", s, flags=re.I)
    if m:
        return canonical("Average", "Wind", m["loc"])
    m = re.match(r"Total (Precipitation|Rainfall|Rain) \((?P<loc>[^)]+)\)$", s, flags=re.I)
    if m:
        return canonical("Total", "Rain", m["loc"])

    # Wind extremes phrased as Name (Wind, Location)
    m = re.match(r"(Max Day|95th Percentile) \(Wind, (?P<loc>[^)]+)\)$", s, flags=re.I)
    if m:
        return canonical(m.group(1), "Wind", m["loc"])

    # Known rain metrics missing type
    rain_names = r"(Max Day|Min Day|Max 5-Day|Min 5-Day|R10mm|R20mm|R95pTOT|CDD)"
    m = re.match(rf"^{rain_names} \((?P<loc>[^)]+)\)$", s, flags=re.I)
    if m:
        return canonical(_norm_5day(m.group(1)), "Rain", m["loc"])

    if "precip" in s.lower():
        m = re.match(r"^(?P<name>[^()]+) \((?P<loc>[^)]+)\)$", s)
        if m:
            return canonical(m["name"], "Rain", m["loc"])

    return s

# ---------- core ----------
def summarise_metrics(df_daily: pd.DataFrame,
                      wet_mm: float,
                      r95_start: int,
                      r95_end: int) -> pd.DataFrame:
    log("Prepare daily frame…")
    df = df_daily.copy()
    df.index = pd.to_datetime(df.index)
    df = df.apply(pd.to_numeric, errors="coerce")

    idx = df.index
    years = np.unique(idx.year)
    log(f"Rows {len(df):,}  years {years.min()}–{years.max()}  columns {list(df.columns)}")

    log("Build seasonal keys…")
    meta = add_season_fields(idx)

    def add_precip(series: pd.Series, region: str):
        log(f"[{region}] Precip metrics start…")
        rows = []
        s = pd.to_numeric(series, errors="coerce")
        r5 = s.rolling(5, min_periods=5).sum()
        years_arr = meta["Season_Year"]; seasons = meta["Season"]
        base = s[(idx.year >= r95_start) & (idx.year <= r95_end) & (s >= wet_mm)]
        r95_thr = pct(base, 0.95) if not base.dropna().empty else np.nan
        dry = (s < wet_mm)

        g = s.groupby([years_arr, seasons], observed=True)
        g5 = r5.groupby([years_arr, seasons], observed=True)
        dryS = dry.groupby([years_arr, seasons], observed=True)

        count_groups = 0
        for (yr, seas), grp in g:
            count_groups += 1
            v = grp.dropna()
            v5 = g5.get_group((yr, seas)).dropna() if (yr, seas) in g5.groups else pd.Series(dtype=float)
            seas_str = str(seas)
            if not v.empty:
                rows += [
                    {"Year": int(yr), "Season": seas_str, "Data Type": canonical("Max Day", "Rain", region), "Value": float(v.max())},
                    {"Year": int(yr), "Season": seas_str, "Data Type": canonical("Min Day", "Rain", region), "Value": float(v.min())},
                    {"Year": int(yr), "Season": seas_str, "Data Type": canonical("Total",   "Rain", region), "Value": float(v.sum())},
                    {"Year": int(yr), "Season": seas_str, "Data Type": canonical("R10mm",   "Rain", region), "Value": float((v >= 10.0).sum())},
                    {"Year": int(yr), "Season": seas_str, "Data Type": canonical("R20mm",   "Rain", region), "Value": float((v >= 20.0).sum())},
                ]
                if not np.isnan(r95_thr):
                    rows.append({"Year": int(yr), "Season": seas_str, "Data Type": canonical("R95pTOT", "Rain", region),
                                 "Value": float(v[v > r95_thr].sum())})
            if not v5.empty:
                rows += [
                    {"Year": int(yr), "Season": seas_str, "Data Type": canonical("Max 5-Day", "Rain", region), "Value": float(v5.max())},
                    {"Year": int(yr), "Season": seas_str, "Data Type": canonical("Min 5-Day", "Rain", region), "Value": float(v5.min())},
                ]
            if (yr, seas) in dryS.groups:
                rows.append({"Year": int(yr), "Season": seas_str, "Data Type": canonical("CDD", "Rain", region),
                             "Value": float(max_consecutive(dryS.get_group((yr, seas))))})

        # Annual
        gY = s.groupby(idx.year); gY5 = r5.groupby(idx.year); dryY = dry.groupby(idx.year)
        for yr, grp in gY:
            v = grp.dropna()
            v5 = gY5.get_group(yr).dropna() if yr in gY5.groups else pd.Series(dtype=float)
            if not v.empty:
                rows += [
                    {"Year": int(yr), "Season": "Annual", "Data Type": canonical("Max Day", "Rain", region), "Value": float(v.max())},
                    {"Year": int(yr), "Season": "Annual", "Data Type": canonical("Min Day", "Rain", region), "Value": float(v.min())},
                    {"Year": int(yr), "Season": "Annual", "Data Type": canonical("Total",   "Rain", region), "Value": float(v.sum())},
                    {"Year": int(yr), "Season": "Annual", "Data Type": canonical("R10mm",   "Rain", region), "Value": float((v >= 10.0).sum())},
                    {"Year": int(yr), "Season": "Annual", "Data Type": canonical("R20mm",   "Rain", region), "Value": float((v >= 20.0).sum())},
                ]
            if not v5.empty:
                rows += [
                    {"Year": int(yr), "Season": "Annual", "Data Type": canonical("Max 5-Day", "Rain", region), "Value": float(v5.max())},
                    {"Year": int(yr), "Season": "Annual", "Data Type": canonical("Min 5-Day", "Rain", region), "Value": float(v5.min())},
                ]
            if yr in dryY.groups:
                rows.append({"Year": int(yr), "Season": "Annual", "Data Type": canonical("CDD", "Rain", region),
                             "Value": float(max_consecutive(dryY.get_group(yr)))})
        log(f"[{region}] Precip metrics done.  Season groups {count_groups}, records {len(rows):,}")
        return rows

    def add_mean(series: pd.Series, region: str, label_typ: str):
        rows = []
        s = pd.to_numeric(series, errors="coerce")
        years_arr = meta["Season_Year"]; seasons = meta["Season"]
        g = s.groupby([years_arr, seasons], observed=True)
        for (yr, seas), grp in g:
            v = grp.dropna()
            if v.empty: continue
            rows.append({"Year": int(yr), "Season": str(seas),
                         "Data Type": canonical("Average", label_typ, region), "Value": float(v.mean())})
        for yr, grp in s.groupby(idx.year):
            v = grp.dropna()
            if not v.empty:
                rows.append({"Year": int(yr), "Season": "Annual",
                             "Data Type": canonical("Average", label_typ, region), "Value": float(v.mean())})
        return rows

    def add_wind_extras(series: pd.Series, region: str):
        rows = []
        s = pd.to_numeric(series, errors="coerce")
        years_arr = meta["Season_Year"]; seasons = meta["Season"]
        g = s.groupby([years_arr, seasons], observed=True)
        for (yr, seas), grp in g:
            v = grp.dropna()
            if v.empty: continue
            rows += [
                {"Year": int(yr), "Season": str(seas),
                 "Data Type": canonical("95th Percentile", "Wind", region), "Value": float(np.percentile(v, 95))},
                {"Year": int(yr), "Season": str(seas),
                 "Data Type": canonical("Max Day", "Wind", region), "Value": float(v.max())},
            ]
        for yr, grp in s.groupby(idx.year):
            v = grp.dropna()
            if not v.empty:
                rows += [
                    {"Year": int(yr), "Season": "Annual",
                     "Data Type": canonical("95th Percentile", "Wind", region), "Value": float(np.percentile(v, 95))},
                    {"Year": int(yr), "Season": "Annual",
                     "Data Type": canonical("Max Day", "Wind", region), "Value": float(v.max())},
                ]
        return rows

    # ---- Temp metrics: means from tas; extremes and 5-day from tasmax if present ----
    def add_temp_metrics(region: str, col_tas: str, col_tasmax: str):
        """
        Adds:
          - Temp (Average, …)     — mean of tas (daily mean)
          - Temp (Max Day, …)     — max day in period (tasmax.max)
          - Temp (Max, …)         — mean of tasmax over period
          - Temp (5-Day Max, …)   — max 5-day rolling mean of tasmax
        Falls back to tas when tasmax columns are absent.
        """
        rows = []
        years_arr = meta["Season_Year"]; seasons = meta["Season"]

        tas = pd.to_numeric(df[col_tas], errors="coerce") if col_tas in df.columns else None
        s = pd.to_numeric(df[col_tasmax], errors="coerce") if col_tasmax in df.columns else None
        if s is None:
            s = tas  # fallback

        # Average from tas
        if tas is not None:
            rows += add_mean(tas, region, "Temp")

        if s is None:
            return rows  # nothing else to add

        r5 = s.rolling(5, min_periods=5).mean()

        # Seasonal
        g = s.groupby([years_arr, seasons], observed=True)
        g5 = r5.groupby([years_arr, seasons], observed=True)
        for (yr, seas), grp in g:
            v = grp.dropna()
            if v.empty: continue
            seas_str = str(seas)
            vmax = float(v.max())           # hottest day in the period
            vmean = float(v.mean())         # mean of tasmax over the period
            rows += [
                {"Year": int(yr), "Season": seas_str, "Data Type": canonical("Max Day", "Temp", region), "Value": vmax},
                {"Year": int(yr), "Season": seas_str, "Data Type": canonical("Max", "Temp", region), "Value": vmean},
            ]
            if (yr, seas) in g5.groups:
                v5 = g5.get_group((yr, seas)).dropna()
                if not v5.empty:
                    rows.append({"Year": int(yr), "Season": seas_str,
                                 "Data Type": canonical("5-Day Max", "Temp", region), "Value": float(v5.max())})

        # Annual
        gY = s.groupby(idx.year); gY5 = r5.groupby(idx.year)
        for yr, grp in gY:
            v = grp.dropna()
            if v.empty: continue
            vmax = float(v.max())
            vmean = float(v.mean())
            rows += [
                {"Year": int(yr), "Season": "Annual", "Data Type": canonical("Max Day", "Temp", region), "Value": vmax},
                {"Year": int(yr), "Season": "Annual", "Data Type": canonical("Max", "Temp", region), "Value": vmean},
            ]
            if yr in gY5.groups:
                v5 = gY5.get_group(yr).dropna()
                if not v5.empty:
                    rows.append({"Year": int(yr), "Season": "Annual",
                                 "Data Type": canonical("5-Day Max", "Temp", region), "Value": float(v5.max())})
        return rows

    recs = []
    # Rain
    recs += add_precip(df["pr_Australia_mm_day"], "Australia")
    recs += add_precip(df["pr_Ravenswood_mm_day"], "Ravenswood")

    # Temp (means from tas; extremes from tasmax if present)
    log("Temperatures…")
    recs += add_temp_metrics("Australia",  "tas_Australia_degC",  "tasmax_Australia_degC")
    recs += add_temp_metrics("Ravenswood", "tas_Ravenswood_degC", "tasmax_Ravenswood_degC")

    # Wind means
    log("Mean wind…")
    recs += add_mean(df["wind_Australia_ms"], "Australia", "Wind")
    recs += add_mean(df["wind_Ravenswood_ms"], "Ravenswood", "Wind")

    # Wind extremes
    log("Wind extremes…")
    recs += add_wind_extras(df["wind_Australia_ms"], "Australia")
    recs += add_wind_extras(df["wind_Ravenswood_ms"], "Ravenswood")

    out = pd.DataFrame.from_records(recs).sort_values(["Year","Season","Data Type"]).reset_index(drop=True)

    # Normalise any legacy labels then assert canonical
    before = out["Data Type"].astype(str)
    out["Data Type"] = out["Data Type"].map(standardise_label)
    fixes = int((before != out["Data Type"]).sum())
    not_canon = sorted(set([t for t in out["Data Type"].unique() if not _CANON_RX.match(str(t))]))
    log(f"Metric labels unified → {fixes} renamed")
    if not_canon:
        log("⚠ Not canonical: " + " | ".join(not_canon))

    log(f"Metrics rows {out.shape[0]:,}")
    return out

# ---------- main ----------
def main():
    folder = os.environ.get("CLIMATE_FOLDER") or FOLDER
    if not os.path.isdir(folder):
        sys.exit(f"Invalid FOLDER: {folder}")

    in_parq = os.path.join(folder, "raw_daily.parquet")
    out_met = os.path.join(folder, "metrics.parquet")
    out_csv = os.path.join(folder, "metrics.csv")

    log(f"Start metrics builder.  Folder: {folder}")
    if not os.path.isfile(in_parq):
        sys.exit(f"Input not found: {in_parq}")

    try:
        size_mb = os.path.getsize(in_parq)/(1024*1024)
        log(f"Read {in_parq}  ({size_mb:.1f} MB)")
        df = pd.read_parquet(in_parq, engine="pyarrow")
        log("Loaded daily frame")

        required = [
            "tas_Australia_degC","tas_Ravenswood_degC",
            "pr_Australia_mm_day","pr_Ravenswood_mm_day",
            "wind_Australia_ms","wind_Ravenswood_ms",
        ]  # tasmax_* are optional
        missing = [c for c in required if c not in df.columns]
        if missing:
            sys.exit(f"Missing columns in input parquet: {missing}")

        log(f"Wet-day threshold {WET_DAY_MM} mm  R95 baseline {R95_BASELINE_START}–{R95_BASELINE_END}")
        metrics = summarise_metrics(df, WET_DAY_MM, R95_BASELINE_START, R95_BASELINE_END)

        log(f"Write {out_met}")
        metrics.to_parquet(out_met, engine="pyarrow", compression="zstd", index=False)
        log(f"Write {out_csv}")
        metrics.to_csv(out_csv, index=False)
        log("✅ Done")
    except SystemExit:
        raise
    except Exception as e:
        log(f"❌ Error: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
