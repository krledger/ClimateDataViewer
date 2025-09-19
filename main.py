#!/usr/bin/env python3
import os, sys, time, socket, webbrowser, subprocess
import pandas as pd

# ===== CONFIG =====
MODE   = "metrics"
PORT   = 8501
#FOLDER = "/Users/ken/Documents/Climate Analysis"
FOLDER = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()

GROUP_KEYS = ["Location","Type","Name","Season","Data Type"]
IDX_COLS_ANNUAL = ["Year"]
IDX_COLS_SEASONAL = ["Year","Season"]

def resolve_folder() -> str:
    if not os.path.isdir(FOLDER):
        sys.exit(f"Invalid folder: {FOLDER}")
    return FOLDER

def _running_under_streamlit() -> bool:
    try:
        import streamlit.runtime as rt
        return rt.exists()
    except Exception:
        return False

def _is_port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.3):
            return True
    except OSError:
        return False

def _launch_self_with_streamlit():
    env = os.environ.copy()
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    script = os.path.abspath(__file__)
    cmd = [sys.executable, "-m", "streamlit", "run", script,
           "--server.port", str(PORT),
           "--browser.gatherUsageStats", "false"]
    print(f"Launching Streamlit → http://localhost:{PORT}", flush=True)
    proc = subprocess.Popen(cmd, env=env)
    for _ in range(40):
        if _is_port_open("127.0.0.1", PORT):
            print(f"Streamlit running at http://localhost:{PORT}", flush=True)
            try: webbrowser.open(f"http://localhost:{PORT}")
            except Exception: pass
            return proc
        time.sleep(0.5)
    print("Streamlit did not start in time.", flush=True)
    return proc

# ---------- UI HELPERS ----------
def tight_label(st, text):
    st.markdown(f'<div style="font-weight:600;font-size:1.05rem;margin:0">{text}</div>', unsafe_allow_html=True)

def checkbox_grid(st, label_html, options, default=None, columns=4, key_prefix="grid"):
    if label_html:
        st.markdown(label_html, unsafe_allow_html=True)
    default = set(default or [])
    sel, cols = [], st.columns(columns)
    for i, opt in enumerate(options):
        with cols[i % columns]:
            if st.checkbox(opt, value=(opt in default), key=f"{key_prefix}_{opt}"):
                sel.append(opt)
    return sel

def chip_multi(st, label_html, options, default=None, columns=2, key_prefix="chip"):
    if label_html:
        st.markdown(label_html, unsafe_allow_html=True)
    default = set(default or [])
    sel, cols = [], st.columns(columns)
    for i, opt in enumerate(options):
        with cols[i % columns]:
            if st.toggle(opt, value=(opt in default), key=f"{key_prefix}_{opt}"):
                sel.append(opt)
    return sel

# ---------- METRICS VIEWER ----------
def run_metrics_viewer():
    if not _running_under_streamlit():
        try: import streamlit  # noqa
        except ImportError:
            sys.exit("pip install streamlit pyarrow altair")
        _launch_self_with_streamlit()
        return

    import streamlit as st
    import altair as alt

    st.set_page_config(page_title="Climate Metrics Viewer", layout="wide")

    st.markdown("""
    <style>
    /* Global sidebar vertical spacing */
    section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] {
      gap: .06rem !important; row-gap: .6rem !important;
    }

    /* Remove extra spacing in BaseWeb form controls (affects radio groups, sliders, toggles) */
    section[data-testid="stSidebar"] div[data-baseweb="form-control"],
    section[data-testid="stSidebar"] div[data-baseweb="form-control-container"],
    section[data-testid="stSidebar"] div[data-baseweb="form-control-content"],
    section[data-testid="stSidebar"] div[data-baseweb="block"] {
      margin: 0 !important; padding: 0 !important;
    }

    /* Collapse radiogroup spacing (Display Mode, Type, Table Interval) */
    section[data-testid="stSidebar"] div[role="radiogroup"],
    section[data-testid="stSidebar"] div[role="radiogroup"] + div {
      margin: 0 !important; padding: 0 !important;
    }

    /* Tighten toggles and checkboxes */
    section[data-testid="stSidebar"] div[data-baseweb="checkbox"],
    section[data-testid="stSidebar"] div[data-baseweb="switch"] {
      margin: 0 !important; padding: 0 !important;
    }

    /* Sliders: Year range and smoothing window */
    section[data-testid="stSidebar"] [data-testid="stSlider"],
    section[data-testid="stSidebar"] [data-testid="stSelectSlider"] {
      margin: 0 !important; padding: 0 !important;
    }

    /* Hide empty help/caption rows under widgets */
    section[data-testid="stSidebar"] div[data-baseweb="form-control-caption"],
    section[data-testid="stSidebar"] [data-testid="stWidgetLabelHelp"] {
      display: none !important; height: 0 !important; margin: 0 !important; padding: 0 !important;
    }

    /* Tighten markdown headings (if you use ###) */
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h4 {
      margin: .05rem 0 !important; line-height: 1.15 !important; padding: 0 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Chart background: default to white unless user set dark theme in settings
    chart_bg = "white" if (st.get_option("theme.base") or "light") == "light" else "transparent"

    def discover_scenarios(base_folder: str):
        return [(name, p, os.path.join(p, fname))
                for name in sorted(os.listdir(base_folder))
                if os.path.isdir((p := os.path.join(base_folder, name)))
                for fname in os.listdir(p)
                if fname.startswith("metrics") and fname.endswith(".parquet")]

    @st.cache_data(show_spinner=False)
    def load_metrics(path: str, mtime: float):
        df = pd.read_parquet(path, engine="pyarrow")
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
        df["Season"] = df.get("Season", "Annual").astype(str)
        df["Data Type"] = df["Data Type"].astype(str)
        df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
        parts = df["Data Type"].str.extract(
            r"^(?P<Type>Temp|Wind|Rain) \((?P<Name>[^,]+), (?P<Location>[^)]+)\)$"
        )
        df = pd.concat([df, parts], axis=1)
        for c in ["Type","Name","Location"]:
            df[c] = df[c].astype(str).str.strip()
        return df.dropna(subset=["Year"]).copy()

    @st.cache_data(show_spinner=False)
    def load_minimal(paths_and_mtimes):
        frames = []
        for label, path, mtime in paths_and_mtimes:
            d = pd.read_parquet(path, engine="pyarrow", columns=["Year","Season","Data Type"])
            parts = d["Data Type"].str.extract(
                r"^(?P<Type>Temp|Wind|Rain) \((?P<Name>[^,]+), (?P<Location>[^)]+)\)$"
            )
            d = pd.concat([d.drop(columns=["Data Type"]), parts], axis=1)
            d["Scenario"] = label
            frames.append(d)
        all_min = pd.concat(frames, ignore_index=True)
        for c in ["Type","Name","Location","Season"]:
            all_min[c] = all_min[c].astype(str).str.strip()
        return all_min.dropna(subset=["Year"])

    def apply_deltas_vs_base(view_in: pd.DataFrame, base_in: pd.DataFrame) -> pd.DataFrame:
        join = ["Year","Season","Data Type","Location","Type","Name"]
        base = base_in[join + ["Value"]].rename(columns={"Value":"Base"})
        out = view_in.merge(base, on=join, how="left", copy=False)
        out["Value"] = out["Value"].where(out["Base"].isna(), out["Value"] - out["Base"])
        return out.drop(columns=["Base"])

    def apply_baseline_from_left_handle(view_in: pd.DataFrame) -> pd.DataFrame:
        if view_in.empty: return view_in
        keys = ["Scenario"] + GROUP_KEYS
        first_year = view_in.groupby(keys, as_index=False)["Year"].min().rename(columns={"Year":"FirstYear"})
        base = (view_in.merge(first_year, on=keys, how="left")
                        .query("Year == FirstYear")[keys + ["Value"]]
                        .rename(columns={"Value":"Baseline"}))
        out = view_in.merge(base, on=keys, how="left")
        out["Value"] = out["Value"].where(out["Baseline"].isna(), out["Value"] - out["Baseline"])
        return out.drop(columns=["Baseline"])

    def apply_smoothing(df_in: pd.DataFrame, window: int) -> pd.DataFrame:
        if window <= 1: return df_in
        df2 = df_in.sort_values(GROUP_KEYS + ["Scenario","Year"])
        if window % 2 == 0: window += 1
        half = max(1, window // 2)
        def _roll(g):
            g = g.sort_values("Year")
            g["Value"] = g["Value"].rolling(window, center=True, min_periods=half).mean()
            return g
        return df2.groupby(["Scenario"] + GROUP_KEYS, group_keys=False).apply(_roll).dropna(subset=["Value"])

    # --- Discover scenarios ---
    base_folder = resolve_folder()
    scenarios = discover_scenarios(base_folder)
    if not scenarios:
        st.error(f"No scenarios found under: {base_folder}")
        st.stop()

    labels = [lbl for (lbl, _, _) in scenarios]
    label_to_metrics = {lbl: m for (lbl, _, m) in scenarios}
    BASE_LABEL = "SSP1-26" if "SSP1-26" in labels else labels[0]
    paths_and_mtimes = [(lbl, label_to_metrics[lbl], os.path.getmtime(label_to_metrics[lbl])) for lbl in labels]
    all_min = load_minimal(paths_and_mtimes)

    # ---------- Sidebar (tight labels = zero margin) ----------
    tight_label(st.sidebar, "Locations")
    loc_candidates = [x for x in ["Ravenswood","Australia"] if x in set(all_min["Location"].unique())] or \
                     sorted(all_min["Location"].dropna().unique())
    loc_default = ["Ravenswood"] if "Ravenswood" in loc_candidates else loc_candidates[:1]
    loc_sel = checkbox_grid(st.sidebar, "", loc_candidates, default=loc_default, columns=3, key_prefix="loc")
    if not loc_sel:
        st.sidebar.warning("Select at least one location."); st.stop()

    tight_label(st.sidebar, "Scenarios")
    scen_sel = checkbox_grid(st.sidebar, "", labels,
                             default=[BASE_LABEL] if BASE_LABEL in labels else [labels[0]],
                             columns=3, key_prefix="scen")
    if not scen_sel:
        st.sidebar.warning("Select at least one scenario."); st.stop()

    tight_label(st.sidebar, "Display Mode")
    mode = st.sidebar.radio("", ["Values", "Baseline (start year)", f"Deltas vs {BASE_LABEL}"],
                            index=0, horizontal=True)
    use_baseline = mode.startswith("Baseline")
    apply_delta = mode.startswith("Deltas")

    tight_label(st.sidebar, "Smoothing")
    smooth = st.sidebar.toggle("Smooth values", value=False)
    smooth_win = st.sidebar.slider("Smoothing window (years)", 3, 21, step=2, value=9)

    tight_label(st.sidebar, "Year range")
    yr_min, yr_max = int(all_min["Year"].min()), int(all_min["Year"].max())
    y0, y1 = st.sidebar.select_slider("",
                                      options=list(range(yr_min + 1, yr_max)),
                                      value=(yr_min + 1, yr_max - 1))

    tight_label(st.sidebar, "Type")
    type_options_all = [t for t in ["Temp","Rain","Wind"] if t in all_min["Type"].unique()] or \
                       sorted(all_min["Type"].unique())
    default_type = "Temp" if "Temp" in type_options_all else type_options_all[0]
    type_sel = st.sidebar.radio("", type_options_all, index=type_options_all.index(default_type), horizontal=True)

    tight_label(st.sidebar, "Metric names")
    avail_for_names = all_min[(all_min["Location"].isin(loc_sel)) & (all_min["Type"] == type_sel)]
    name_options = sorted(avail_for_names["Name"].dropna().unique())
    if type_sel == "Temp":
        hint = ["Average","Max","Max Day","5-Day Max","Max 5-Day Average"]
        name_options = sorted(name_options, key=lambda n: (hint.index(n) if n in hint else 99, n))
    default_names = ["Average"] if "Average" in name_options else (name_options[:1] if name_options else [])
    name_sel = chip_multi(st.sidebar, "", name_options, default=default_names, columns=2, key_prefix="metricchip")
    if not name_sel:
        st.sidebar.warning("Select at least one metric."); st.stop()

    tight_label(st.sidebar, "Seasons")
    seasons_all = ["Annual","DJF","MAM","JJA","SON"]
    have_seasons = [s for s in seasons_all if s in all_min["Season"].unique()]
    default_seasons = ["Annual"] if "Annual" in have_seasons else have_seasons
    season_sel = checkbox_grid(st.sidebar, "", have_seasons, default=default_seasons, columns=5, key_prefix="seas")
    if not season_sel:
        st.sidebar.warning("Select at least one season."); st.stop()

    tight_label(st.sidebar, "Table interval")
    table_interval = st.sidebar.radio("", [1, 2, 5, 10], index=0, horizontal=True)

    # ---------- Title ----------
    st.title("Climate Metrics Viewer")
    st.caption(f"Locations: {', '.join(loc_sel)}  •  Scenarios: {', '.join(scen_sel)}  •  Years: {y0}–{y1}  •  Display Mode: {mode}")

    # ---------- Load + filter ----------
    dfs = []
    for lbl in scen_sel:
        p = label_to_metrics[lbl]
        dfi = load_metrics(p, os.path.getmtime(p)).copy()
        dfi["Scenario"] = lbl
        dfs.append(dfi)
    df_all = pd.concat(dfs, ignore_index=True)

    base_df = load_metrics(label_to_metrics[BASE_LABEL],
                           os.path.getmtime(label_to_metrics[BASE_LABEL])).copy()
    base_df["Scenario"] = BASE_LABEL

    mask = (
        df_all["Year"].between(y0, y1) &
        df_all["Location"].isin(loc_sel) &
        df_all["Season"].isin(season_sel) &
        (df_all["Type"] == type_sel) &
        df_all["Name"].isin(name_sel)
    )
    pre_transform_view = df_all.loc[mask].copy()
    view = pre_transform_view.copy()

    # Transform order (smooth→baseline so baseline starts at 0 after smoothing)
    if use_baseline and smooth:
        view = apply_smoothing(view, smooth_win)
        view = apply_baseline_from_left_handle(view)
    else:
        if use_baseline:
            view = apply_baseline_from_left_handle(view)
        if apply_delta:
            view = apply_deltas_vs_base(view, base_df)
        if smooth:
            view = apply_smoothing(view, smooth_win)
    if smooth:
        view = view.dropna(subset=["Value"])

    idx_cols = IDX_COLS_ANNUAL if season_sel == ["Annual"] or not season_sel else IDX_COLS_SEASONAL

   # Chart
    if not view.empty:
        st.subheader("Chart")
        plot = view[idx_cols + ["Data Type","Value","Location","Name","Scenario"]].rename(columns={"Data Type":"Metric"})
        if idx_cols == IDX_COLS_ANNUAL:
            plot["X"] = plot["Year"].astype(int)
            # lock domain to slider to keep left origin in sync
            x_enc = alt.X("X:Q", title="Year", axis=alt.Axis(format='d'),
                          scale=alt.Scale(domain=[int(y0), int(y1)]))
        else:
            season_order = ["DJF","MAM","JJA","SON"]
            plot["Season"] = pd.Categorical(plot["Season"], categories=season_order, ordered=True)
            plot = plot.sort_values(["Year","Season"])
            plot["X"] = plot["Year"].astype(int).astype(str) + "-" + plot["Season"].astype(str)
            x_enc = alt.X("X:N", title="Year–Season", sort=list(plot["X"].unique()))
        sel = alt.selection_point(fields=["Metric","Scenario"], bind="legend")
        if not view.empty:
            plot = view[idx_cols + ["Data Type", "Value", "Location", "Name", "Scenario"]].rename(
                columns={"Data Type": "Metric"})
            if idx_cols == IDX_COLS_ANNUAL:
                plot["X"] = plot["Year"].astype(int)
                x_enc = alt.X("X:Q", title="Year", axis=alt.Axis(format='d', labelFontSize=14, titleFontSize=16),
                              scale=alt.Scale(domain=[int(y0), int(y1)]))
            else:
                season_order = ["DJF", "MAM", "JJA", "SON"]
                plot["Season"] = pd.Categorical(plot["Season"], categories=season_order, ordered=True)
                plot = plot.sort_values(["Year", "Season"])
                plot["X"] = plot["Year"].astype(int).astype(str) + "-" + plot["Season"].astype(str)
                x_enc = alt.X("X:N", title="Year–Season",
                              axis=alt.Axis(labelFontSize=14, titleFontSize=16),
                              sort=list(plot["X"].unique()))
            sel = alt.selection_point(fields=["Metric", "Scenario"], bind="legend")

            chart = (
                alt.Chart(plot)
                .mark_line(point=True, strokeWidth=3)  # thicker lines (default ~1.5)
                .encode(
                    x=x_enc,
                    y=alt.Y("Value:Q", title="Value", scale=alt.Scale(zero=False),
                            axis=alt.Axis(labelFontSize=16, titleFontSize=18)),
                    color=alt.Color(
                        "Scenario:N",
                        title="Scenario",
                        legend=alt.Legend(
                            labelFontSize=16,
                            titleFontSize=18,
                            labelLimit=500
                        )
                    ),
                    strokeDash=alt.StrokeDash(
                        "Metric:N",
                        title="Metric",
                        legend=alt.Legend(
                            labelFontSize=16,
                            titleFontSize=18,
                            labelLimit=300
                        )
                    ),
                    tooltip=[alt.Tooltip("Scenario:N"),
                             alt.Tooltip("Metric:N"),
                             alt.Tooltip("Value:Q", format=",.3f"),
                             alt.Tooltip("X:N", title="Period"),
                             alt.Tooltip("Location:N"),
                             alt.Tooltip("Name:N")],
                    opacity=alt.condition(sel, alt.value(1), alt.value(0.25))
                )
                .add_params(sel)
                .properties(height=500, width=1500, background=chart_bg)
            )

            st.altair_chart(chart, use_container_width=False)

    # Baseline preview row
    if use_baseline and not pre_transform_view.empty:
        baseline_slice = pre_transform_view[pre_transform_view["Year"] == y0].copy()
        idx_cols_base = idx_cols
        if len(scen_sel) > 1:
            baseline_table = baseline_slice.pivot_table(index=idx_cols_base,
                                                        columns=["Data Type", "Scenario"],
                                                        values="Value", aggfunc="first").sort_index()
        else:
            baseline_table = baseline_slice.pivot_table(index=idx_cols_base,
                                                        columns="Data Type",
                                                        values="Value", aggfunc="first").sort_index()
        st.subheader(f"Baseline values (Year = {y0})")
        st.dataframe(baseline_table, use_container_width=True)

    # Table
    table_view = view.copy()
    if table_interval > 1 and not table_view.empty and "Year" in table_view.columns:
        anchor = int(table_view["Year"].min())
        table_view = table_view[((table_view["Year"].astype(int) - anchor) % table_interval) == 0]

    if len(scen_sel) > 1:
        table = table_view.pivot_table(index=idx_cols,
                                       columns=["Data Type","Scenario"],
                                       values="Value", aggfunc="first").sort_index()
    else:
        table = table_view.pivot_table(index=idx_cols,
                                       columns="Data Type",
                                       values="Value", aggfunc="first").sort_index()

    st.subheader("Values")
    st.dataframe(table, use_container_width=True, height=520)

if __name__ == "__main__":
    if MODE.lower() == "metrics":
        run_metrics_viewer()
    else:
        sys.exit('Set MODE = "metrics"')
