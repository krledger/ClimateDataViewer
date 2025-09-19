#!/usr/bin/env python3
import os, sys, time, socket, webbrowser, subprocess
import pandas as pd

# ===== CONFIG =====
MODE   = "metrics"
PORT   = 8501
import os, sys

# Auto-detect the folder where the exe/script lives
FOLDER = os.path.dirname(os.path.abspath(sys.executable if getattr(sys, "frozen", False) else __file__))
#FOLDER = "/Users/ken/Documents/Climate Analysis"

# ==================

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

    def discover_scenarios(base_folder: str):
        return [(name, p, os.path.join(p, fname))
                for name in sorted(os.listdir(base_folder))
                if os.path.isdir((p := os.path.join(base_folder, name)))
                for fname in os.listdir(p)
                if fname.startswith("metrics") and fname.endswith(".parquet")]

    @st.cache_data(show_spinner=False)
    def load_metrics(path: str, mtime: float):
        # Load all columns to preserve labels, then derive helpers
        df = pd.read_parquet(path, engine="pyarrow")
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
        df["Season"] = df.get("Season", "Annual").astype(str)
        df["Data Type"] = df["Data Type"].astype(str)
        df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
        parts = df["Data Type"].str.extract(r"^(?P<Type>Temp|Wind|Rain) \((?P<Name>[^,]+), (?P<Location>[^)]+)\)$")
        df = pd.concat([df, parts], axis=1)
        for c in ["Type","Name","Location"]:
            df[c] = df[c].astype(str).str.strip()
        return df.dropna(subset=["Year"]).copy()

    def apply_deltas(view: pd.DataFrame, base_df: pd.DataFrame) -> pd.DataFrame:
        # Join on keys that uniquely define a metric row across scenarios
        join_keys = ["Year","Season","Data Type"]
        base = base_df[join_keys + ["Value"]].rename(columns={"Value":"Base"})
        out = view.merge(base, on=join_keys, how="left", copy=False)
        # Leave Value unchanged if Base is missing, else compute delta
        out["Value"] = out["Value"].where(out["Base"].isna(), out["Value"] - out["Base"])
        return out.drop(columns=["Base"])

    def apply_smoothing(df_in: pd.DataFrame, window: int) -> pd.DataFrame:
        # Centred moving average along Year within each series.  Labels unchanged.
        if window <= 1: return df_in
        df2 = df_in.sort_values(GROUP_KEYS + ["Year"])
        # Force odd window for symmetry
        if window % 2 == 0: window += 1
        half = max(1, window // 2)
        def _roll(g):
            g = g.sort_values("Year")
            g["Value"] = g["Value"].rolling(window, center=True, min_periods=half).mean()
            return g
        return df2.groupby(GROUP_KEYS, group_keys=False).apply(_roll).dropna(subset=["Value"])

    # --- Scenario discovery and base setup ---
    base_folder = resolve_folder()
    scenarios = discover_scenarios(base_folder)
    if not scenarios:
        st.error(f"No scenarios found under: {base_folder}")
        st.stop()

    labels = [lbl for (lbl, _, _) in scenarios]
    label_to_metrics = {lbl: m for (lbl, _, m) in scenarios}
    default_label = "SSP1-26" if "SSP1-26" in labels else labels[0]
    BASE_LABEL = "SSP1-26"

    scenario_label = st.sidebar.selectbox("Scenario", options=labels,
                                          index=labels.index(default_label))
    metrics_path = label_to_metrics[scenario_label]

    if st.sidebar.button("Reload metrics"):
        st.cache_data.clear()
        st.session_state.clear()
        st.rerun()

    df = load_metrics(metrics_path, os.path.getmtime(metrics_path))
    base_df = load_metrics(label_to_metrics[BASE_LABEL],
                           os.path.getmtime(label_to_metrics[BASE_LABEL]))

    # Slider range with global trim ±1 year
    raw_min, raw_max = int(df["Year"].min()), int(df["Year"].max())
    metrics_min, metrics_max = raw_min + 1, raw_max - 1
    if metrics_min >= metrics_max:
        metrics_min, metrics_max = raw_min, raw_max

    # Top controls
    apply_delta = st.sidebar.toggle(f"Show deltas vs {BASE_LABEL}", value=False)
    smooth = st.sidebar.toggle("Smooth values", value=False)
    smooth_win = st.sidebar.slider("Smoothing window (years)", 3, 21, step=2, value=9)

    y0, y1 = st.sidebar.select_slider("Year range",
                                      options=list(range(metrics_min, metrics_max + 1)),
                                      value=(metrics_min, metrics_max))

    # Filters
    loc_options = [x for x in ["Ravenswood","Australia"] if x in df["Location"].unique()]
    loc_options = loc_options or sorted(df["Location"].dropna().unique())
    loc_sel = st.sidebar.multiselect("Location", loc_options,
                                     default=["Ravenswood"] if "Ravenswood" in loc_options else loc_options[:1])

    type_options = [t for t in ["Temp","Rain","Wind"] if t in df["Type"].unique()] or sorted(df["Type"].unique())
    type_sel = st.sidebar.selectbox("Type", type_options,
                                    index=type_options.index("Temp") if "Temp" in type_options else 0)

    df_scope = df[df["Location"].isin(loc_sel) & (df["Type"] == type_sel)]
    name_options = sorted(df_scope["Name"].dropna().unique())
    if type_sel == "Temp":
        hint = ["Average","Max","Max Day","5-Day Max","Max 5-Day Average"]
        name_options = sorted(name_options, key=lambda n: (hint.index(n) if n in hint else 99, n))
    name_sel = st.sidebar.multiselect("Metric name(s)", options=name_options, default=name_options)

    seasons_all = ["Annual","DJF","MAM","JJA","SON"]
    have_seasons = [s for s in seasons_all if s in df["Season"].unique()]
    season_sel = st.sidebar.multiselect("Seasons", have_seasons,
                                       default=[s for s in have_seasons if s != "Annual"] or have_seasons)

    # Filter, trim, transform
    mask = (
        df["Year"].between(y0, y1) &
        df["Location"].isin(loc_sel) &
        df["Season"].isin(season_sel) &
        (df["Type"] == type_sel) &
        df["Name"].isin(name_sel)
    )
    view = df.loc[mask].copy()
    if not view.empty:
        view = view[view["Year"].between(view["Year"].min() + 1, view["Year"].max() - 1)]

    if apply_delta:
        if scenario_label == BASE_LABEL:
            view = view.copy()
            view["Value"] = 0.0
            st.caption(f"Deltas vs {BASE_LABEL}: all values are 0.")
        else:
            view = apply_deltas(view, base_df)
            st.caption(f"Δ relative to {BASE_LABEL} applied.")

    if smooth:
        view = apply_smoothing(view, smooth_win)
        st.caption(f"⚠️ {smooth_win}-year centred moving average applied.")

    # Table
    idx_cols = IDX_COLS_ANNUAL if season_sel == ["Annual"] or not season_sel else IDX_COLS_SEASONAL
    table = view.pivot_table(index=idx_cols, columns="Data Type", values="Value", aggfunc="first").sort_index()

    import streamlit as st  # ensure in scope for PyInstaller bundling edge cases
    st.title("Climate Metrics Viewer")
    st.caption(f"Base folder: {resolve_folder()}  •  Scenario: {scenario_label}")

    st.subheader("Values")
    st.dataframe(table, use_container_width=True, height=520)

    # Chart
    if not view.empty:
        import altair as alt  # ensure in scope
        st.subheader("Chart")
        plot = view[idx_cols + ["Data Type","Value","Location","Name"]].rename(columns={"Data Type":"Metric"})
        if idx_cols == IDX_COLS_ANNUAL:
            plot["X"] = plot["Year"].astype(int)
            x_enc = alt.X("X:Q", title="Year")
        else:
            season_order = ["DJF","MAM","JJA","SON"]
            plot["Season"] = pd.Categorical(plot["Season"], categories=season_order, ordered=True)
            plot = plot.sort_values(["Year","Season"])
            plot["X"] = plot["Year"].astype(int).astype(str) + "-" + plot["Season"].astype(str)
            x_enc = alt.X("X:N", title="Year–Season", sort=list(plot["X"].unique()))
        sel = alt.selection_point(fields=["Metric"], bind="legend")
        chart = (alt.Chart(plot)
                 .mark_line(point=True)
                 .encode(x=x_enc,
                         y=alt.Y("Value:Q", title="Value", scale=alt.Scale(zero=False)),
                         color=alt.Color("Metric:N", title="Metric"),
                         tooltip=[alt.Tooltip("Metric:N"),
                                  alt.Tooltip("Value:Q", format=",.3f"),
                                  alt.Tooltip("X:N", title="Period"),
                                  alt.Tooltip("Location:N"),
                                  alt.Tooltip("Name:N")],
                         opacity=alt.condition(sel, alt.value(1), alt.value(0.2)))
                 .add_params(sel).properties(height=360).interactive())
        st.altair_chart(chart, use_container_width=True)

    # Download  (choose smoothed or current view)
    dl_smoothed = st.sidebar.toggle("Download smoothed values", value=False)
    csv_df = table if dl_smoothed else view.pivot_table(index=idx_cols, columns="Data Type", values="Value", aggfunc="first").sort_index()
    st.download_button("Download filtered table (CSV)", data=csv_df.to_csv(),
                       file_name=f"{scenario_label}_metrics_filtered.csv", mime="text/csv")

    st.caption(f"{len(view):,} rows filtered • {table.shape[1]} metrics shown")

if __name__ == "__main__":
    if MODE.lower() == "metrics":
        run_metrics_viewer()
    else:
        sys.exit('Set MODE = "metrics"')
