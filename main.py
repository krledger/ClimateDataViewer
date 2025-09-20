#!/usr/bin/env python3
# Climate Metrics Viewer (Streamlit) — single-file app
# - Auth gate with narrow page width + disclaimer
# - Metrics browser with Altair charts and pivoted tables

import os, sys, time, socket, webbrowser, subprocess, re
import pandas as pd

# ============================== CONFIG ========================================
MODE = "metrics"
PORT = 8501
SECURE = True                             # set False to bypass login
ALLOWED_CODES = {"Chaos123", "Decision456"}   # invite codes
FOLDER = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()

GROUP_KEYS = ["Location", "Type", "Name", "Season", "Data Type"]
IDX_COLS_ANNUAL = ["Year"]
IDX_COLS_SEASONAL = ["Year", "Season"]


# ============================ RUNTIME HELPERS =================================
def resolve_folder() -> str:
    """Validate and return the base data folder."""
    if not os.path.isdir(FOLDER):
        sys.exit(f"Invalid folder: {FOLDER}")
    return FOLDER


def _running_under_streamlit() -> bool:
    """Detect if code is running via `streamlit run`."""
    try:
        import streamlit.runtime as rt
        return rt.exists()
    except Exception:
        return False


def _is_port_open(host: str, port: int) -> bool:
    """Check if a localhost TCP port is open."""
    try:
        with socket.create_connection((host, port), timeout=0.3):
            return True
    except OSError:
        return False


def _launch_self_with_streamlit():
    """Re-run this script via `streamlit run` and open the browser."""
    env = os.environ.copy()
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    script = os.path.abspath(__file__)
    cmd = [sys.executable, "-m", "streamlit", "run", script, "--server.port", str(PORT), "--browser.gatherUsageStats", "false"]
    print(f"Launching Streamlit → http://localhost:{PORT}", flush=True)
    proc = subprocess.Popen(cmd, env=env)
    for _ in range(40):
        if _is_port_open("127.0.0.1", PORT):
            print(f"Streamlit running at http://localhost:{PORT}", flush=True)
            try:
                webbrowser.open(f"http://localhost:{PORT}")
            except Exception:
                pass
            return proc
        time.sleep(0.5)
    print("Streamlit did not start in time.", flush=True)
    return proc


# ============================== UI HELPERS ====================================
def tight_label(container, text: str):
    """Compact bold label for sidebar sections."""
    container.markdown(
        f'<div style="font-weight:600;font-size:1.05rem;margin:0">{text}</div>',
        unsafe_allow_html=True,
    )


def _slug(s: str) -> str:
    """URL-safe key fragment."""
    return re.sub(r"[^a-z0-9]+", "-", str(s).lower()).strip("-")


def _dedupe_preserve_order(items):
    """De-dupe while preserving first occurrence order."""
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def checkbox_grid(container, label_html, options, default=None, columns=4, key_prefix="grid", namespace=None):
    """Multi-column grid of checkboxes; returns selected list.  Keys are unique via index + slug + namespace."""
    if label_html:
        container.markdown(label_html, unsafe_allow_html=True)
    opts = _dedupe_preserve_order(options or [])
    default = set(default or [])
    cols = container.columns(columns)
    selected = []
    for i, opt in enumerate(opts):
        key = f"{namespace + '_' if namespace else ''}{key_prefix}_{i}_{_slug(opt)}"
        with cols[i % columns]:
            if container.checkbox(str(opt), value=(opt in default), key=key):
                selected.append(opt)
    return selected


def chip_multi(container, label_html, options, default=None, columns=2, key_prefix="chip", namespace=None):
    """Toggle chips (using st.toggle) in a grid; returns selected list.  Keys are unique via index + slug + namespace."""
    if label_html:
        container.markdown(label_html, unsafe_allow_html=True)
    opts = _dedupe_preserve_order(options or [])
    default = set(default or [])
    cols = container.columns(columns)
    selected = []
    for i, opt in enumerate(opts):
        key = f"{namespace + '_' if namespace else ''}{key_prefix}_{i}_{_slug(opt)}"
        with cols[i % columns]:
            if container.toggle(str(opt), value=(opt in default), key=key):
                selected.append(opt)
    return selected


# ============================ METRICS VIEWER ==================================
def run_metrics_viewer():
    """Main Streamlit app entrypoint."""
    if not _running_under_streamlit():
        try:
            import streamlit  # noqa: F401
        except ImportError:
            sys.exit("pip install streamlit pyarrow altair")
        _launch_self_with_streamlit()
        return

    import streamlit as st
    import altair as alt

    # --- Session auth flag (default false) ---
    if "invite_authenticated" not in st.session_state:
        st.session_state.invite_authenticated = False

    # --- Decide layout ONCE, then set page config ONCE ---
    layout_mode = "centered" if (SECURE and not st.session_state.invite_authenticated) else "wide"
    st.set_page_config(page_title="Climate Metrics Viewer", layout=layout_mode)

    # --- Narrow page CSS only during login (800px) ---
    if SECURE and not st.session_state.invite_authenticated:
        st.markdown(
            """
            <style>
              .block-container { max-width: 800px; }
            </style>
            """,
            unsafe_allow_html=True,
        )

        # Disclaimer (read-only)
        disclaimer = (
            "This application is provided on an as-is basis.  Use of any content or data is at your own risk.  "
            "No audit, verification or validation has been performed.  Nothing here is advice, recommendation or "
            "a substitute for professional judgement.  No warranty is given as to accuracy, completeness or "
            "fitness for any purpose.  The authors and operators accept no liability for any loss or damage "
            "arising from use, misuse or reliance.  By proceeding you accept these terms."
        )
        st.markdown(
            f"""
            <div style="border:1px solid #ccc;border-radius:6px;background:#f9f9f9;
                        padding:12px;margin-bottom:20px;color:black;">
              {disclaimer}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Login box
        with st.container(border=True):
            st.subheader("Restricted access")
            code = st.text_input("Enter invite code", type="password", key="login_invite_code")
            if st.button("Unlock", key="login_unlock_btn"):
                if code in ALLOWED_CODES:
                    st.session_state.invite_authenticated = True
                    st.success("Access granted.  Loading dashboard…")
                    st.rerun()
                else:
                    st.error("Invalid login")
        st.stop()

    # ====================== APP (AUTHENTICATED) UI TUNING =====================
    st.markdown(
        """
        <style>
        /* Sidebar vertical spacing + compact BaseWeb controls */
        section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] { gap:.06rem !important; row-gap:.6rem !important; }
        section[data-testid="stSidebar"] div[data-baseweb="form-control"],
        section[data-testid="stSidebar"] div[data-baseweb="form-control-container"],
        section[data-testid="stSidebar"] div[data-baseweb="form-control-content"],
        section[data-testid="stSidebar"] div[data-baseweb="block"] { margin:0 !important; padding:0 !important; }
        section[data-testid="stSidebar"] div[role="radiogroup"],
        section[data-testid="stSidebar"] div[role="radiogroup"] + div { margin:0 !important; padding:0 !important; }
        section[data-testid="stSidebar"] div[data-baseweb="checkbox"],
        section[data-testid="stSidebar"] div[data-baseweb="switch"] { margin:0 !important; padding:0 !important; }
        section[data-testid="stSidebar"] [data-testid="stSlider"],
        section[data-testid="stSidebar"] [data-testid="stSelectSlider"] { margin:0 !important; padding:0 !important; }
        section[data-testid="stSidebar"] div[data-baseweb="form-control-caption"],
        section[data-testid="stSidebar"] [data-testid="stWidgetLabelHelp"] { display:none !important; height:0 !important; margin:0 !important; padding:0 !important; }
        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3,
        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h4 { margin:.05rem 0 !important; line-height:1.15 !important; padding:0 !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Theme-aware chart background
    chart_bg = "white" if (st.get_option("theme.base") or "light") == "light" else "transparent"

    # --------------------------- Data discovery/load ---------------------------
    def discover_scenarios(base_folder: str):
        """Find (label, label_path, metrics_path) for each scenario folder containing a metrics parquet."""
        return [
            (name, p, os.path.join(p, fname))
            for name in sorted(os.listdir(base_folder))
            if os.path.isdir((p := os.path.join(base_folder, name)))
            for fname in os.listdir(p)
            if fname.startswith("metrics") and fname.endswith(".parquet")
        ]

    @st.cache_data(show_spinner=False)
    def load_metrics(path: str, mtime: float):
        """Load a single metrics parquet and parse 'Data Type' into Type/Name/Location."""
        df = pd.read_parquet(path, engine="pyarrow")
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
        df["Season"] = df.get("Season", "Annual").astype(str)
        df["Data Type"] = df["Data Type"].astype(str)
        df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
        parts = df["Data Type"].str.extract(r"^(?P<Type>Temp|Wind|Rain) \((?P<Name>[^,]+), (?P<Location>[^)]+)\)$")
        df = pd.concat([df, parts], axis=1)
        for c in ["Type", "Name", "Location"]:
            df[c] = df[c].astype(str).str.strip()
        return df.dropna(subset=["Year"]).copy()

    @st.cache_data(show_spinner=False)
    def load_minimal(paths_and_mtimes):
        """Lightweight union of scenario metadata for fast sidebar options."""
        frames = []
        for label, path, mtime in paths_and_mtimes:
            d = pd.read_parquet(path, engine="pyarrow", columns=["Year", "Season", "Data Type"])
            parts = d["Data Type"].str.extract(r"^(?P<Type>Temp|Wind|Rain) \((?P<Name>[^,]+), (?P<Location>[^)]+)\)$")
            d = pd.concat([d.drop(columns=["Data Type"]), parts], axis=1)
            d["Scenario"] = label
            frames.append(d)
        all_min = pd.concat(frames, ignore_index=True)
        for c in ["Type", "Name", "Location", "Season"]:
            all_min[c] = all_min[c].astype(str).str.strip()
        return all_min.dropna(subset=["Year"])

    # --------------------------- Transform utilities --------------------------
    def apply_deltas_vs_base(view_in: pd.DataFrame, base_in: pd.DataFrame) -> pd.DataFrame:
        """Subtract base scenario values (by Year/Season/Data Type/Location/Type/Name)."""
        join = ["Year", "Season", "Data Type", "Location", "Type", "Name"]
        base = base_in[join + ["Value"]].rename(columns={"Value": "Base"})
        out = view_in.merge(base, on=join, how="left", copy=False)
        out["Value"] = out["Value"].where(out["Base"].isna(), out["Value"] - out["Base"])
        return out.drop(columns=["Base"])

    def apply_baseline_from_left_handle(view_in: pd.DataFrame) -> pd.DataFrame:
        """Subtract the first available-year value within each (Scenario + group) so first year = 0."""
        if view_in.empty:
            return view_in
        keys = ["Scenario"] + GROUP_KEYS
        first_year = view_in.groupby(keys, as_index=False)["Year"].min().rename(columns={"Year": "FirstYear"})
        base = (
            view_in.merge(first_year, on=keys, how="left")
            .query("Year == FirstYear")[keys + ["Value"]]
            .rename(columns={"Value": "Baseline"})
        )
        out = view_in.merge(base, on=keys, how="left")
        out["Value"] = out["Value"].where(out["Baseline"].isna(), out["Value"] - out["Baseline"])
        return out.drop(columns=["Baseline"])

    def apply_smoothing(df_in: pd.DataFrame, window: int) -> pd.DataFrame:
        """Centered rolling-mean smoothing by group (odd window; min periods = window//2)."""
        if window <= 1:
            return df_in
        df2 = df_in.sort_values(GROUP_KEYS + ["Scenario", "Year"])
        if window % 2 == 0:
            window += 1
        half = max(1, window // 2)

        def _roll(g):
            g = g.sort_values("Year")
            g["Value"] = g["Value"].rolling(window, center=True, min_periods=half).mean()
            return g

        return df2.groupby(["Scenario"] + GROUP_KEYS, group_keys=False).apply(_roll).dropna(subset=["Value"])

    # ------------------------------- Discover --------------------------------
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

    # ============================== SIDEBAR ===================================
    ns = "metrics"  # namespace for unique keys on this screen

    tight_label(st.sidebar, "Locations")
    loc_candidates = [x for x in ["Ravenswood", "Australia"] if x in set(all_min["Location"].unique())] or \
                     sorted(all_min["Location"].dropna().unique())
    loc_candidates = _dedupe_preserve_order(loc_candidates)
    loc_default = ["Ravenswood"] if "Ravenswood" in loc_candidates else loc_candidates[:1]
    loc_sel = checkbox_grid(st.sidebar, "", loc_candidates, default=loc_default, columns=3, key_prefix="loc", namespace=ns)
    if not loc_sel:
        st.sidebar.warning("Select at least one location.")
        st.stop()

    tight_label(st.sidebar, "Scenarios")
    scen_sel = checkbox_grid(
        st.sidebar, "", labels,
        default=[BASE_LABEL] if BASE_LABEL in labels else [labels[0]],
        columns=3, key_prefix="scen", namespace=ns
    )
    if not scen_sel:
        st.sidebar.warning("Select at least one scenario.")
        st.stop()

    tight_label(st.sidebar, "Display Mode")
    mode = st.sidebar.radio("", ["Values", "Baseline (start year)", f"Deltas vs {BASE_LABEL}"], index=0, horizontal=True, key=f"{ns}_mode")
    use_baseline = mode.startswith("Baseline")
    apply_delta = mode.startswith("Deltas")

    tight_label(st.sidebar, "Smoothing")
    smooth = st.sidebar.toggle("Smooth values", value=False, key=f"{ns}_smooth")
    smooth_win = st.sidebar.slider("Smoothing window (years)", 3, 21, step=2, value=9, key=f"{ns}_smooth_win")

    tight_label(st.sidebar, "Year range")
    yr_min, yr_max = int(all_min["Year"].min()), int(all_min["Year"].max())
    y0, y1 = st.sidebar.select_slider(
        "", options=list(range(yr_min + 1, yr_max)),
        value=(yr_min + 1, yr_max - 1),
        key=f"{ns}_yr"
    )

    tight_label(st.sidebar, "Type")
    type_options_all = [t for t in ["Temp", "Rain", "Wind"] if t in all_min["Type"].unique()] or sorted(all_min["Type"].unique())
    default_type = "Temp" if "Temp" in type_options_all else type_options_all[0]
    type_sel = st.sidebar.radio("", type_options_all, index=type_options_all.index(default_type), horizontal=True, key=f"{ns}_type")

    tight_label(st.sidebar, "Metric names")
    avail_for_names = all_min[(all_min["Location"].isin(loc_sel)) & (all_min["Type"] == type_sel)]
    name_options = sorted(avail_for_names["Name"].dropna().unique())
    if type_sel == "Temp":
        hint = ["Average", "Max", "Max Day", "5-Day Max", "Max 5-Day Average"]
        name_options = sorted(name_options, key=lambda n: (hint.index(n) if n in hint else 99, n))
    default_names = ["Average"] if "Average" in name_options else (name_options[:1] if name_options else [])
    name_sel = chip_multi(st.sidebar, "", name_options, default=default_names, columns=2, key_prefix="metricchip", namespace=ns)
    if not name_sel:
        st.sidebar.warning("Select at least one metric.")
        st.stop()

    tight_label(st.sidebar, "Seasons")
    seasons_all = ["Annual", "DJF", "MAM", "JJA", "SON"]
    have_seasons = [s for s in seasons_all if s in all_min["Season"].unique()]
    default_seasons = ["Annual"] if "Annual" in have_seasons else have_seasons
    season_sel = checkbox_grid(st.sidebar, "", have_seasons, default=default_seasons, columns=5, key_prefix="seas", namespace=ns)
    if not season_sel:
        st.sidebar.warning("Select at least one season.")
        st.stop()

    tight_label(st.sidebar, "Table interval")
    table_interval = st.sidebar.radio("", [1, 2, 5, 10], index=0, horizontal=True, key=f"{ns}_tblint")

    # =============================== CONTENT ==================================
    st.title("Climate Metrics Viewer")
    st.caption(f"Locations: {', '.join(loc_sel)}  •  Scenarios: {', '.join(scen_sel)}  •  Years: {y0}–{y1}  •  Display Mode: {mode}")

    # Load & filter
    dfs = []
    for lbl in scen_sel:
        p = label_to_metrics[lbl]
        dfi = load_metrics(p, os.path.getmtime(p)).copy()
        dfi["Scenario"] = lbl
        dfs.append(dfi)
    df_all = pd.concat(dfs, ignore_index=True)

    base_df = load_metrics(label_to_metrics[BASE_LABEL], os.path.getmtime(label_to_metrics[BASE_LABEL])).copy()
    base_df["Scenario"] = BASE_LABEL

    mask = (
        df_all["Year"].between(y0, y1)
        & df_all["Location"].isin(loc_sel)
        & df_all["Season"].isin(season_sel)
        & (df_all["Type"] == type_sel)
        & df_all["Name"].isin(name_sel)
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
        plot = view[idx_cols + ["Data Type", "Value", "Location", "Name", "Scenario"]].rename(columns={"Data Type": "Metric"})
        if idx_cols == IDX_COLS_ANNUAL:
            plot["X"] = plot["Year"].astype(int)
            x_enc = alt.X("X:Q", title="Year", axis=alt.Axis(format="d", labelFontSize=14, titleFontSize=16),
                          scale=alt.Scale(domain=[int(y0), int(y1)]))
        else:
            season_order = ["DJF", "MAM", "JJA", "SON"]
            plot["Season"] = pd.Categorical(plot["Season"], categories=season_order, ordered=True)
            plot = plot.sort_values(["Year", "Season"])
            plot["X"] = plot["Year"].astype(int).astype(str) + "-" + plot["Season"].astype(str)
            x_enc = alt.X("X:N", title="Year–Season", axis=alt.Axis(labelFontSize=14, titleFontSize=16),
                          sort=list(plot["X"].unique()))
        sel = alt.selection_point(fields=["Metric", "Scenario"], bind="legend")

        chart = (
            alt.Chart(plot)
            .mark_line(point=True, strokeWidth=3)
            .encode(
                x=x_enc,
                y=alt.Y("Value:Q", title="Value", scale=alt.Scale(zero=False),
                        axis=alt.Axis(labelFontSize=16, titleFontSize=18)),
                color=alt.Color("Scenario:N", title="Scenario",
                                legend=alt.Legend(labelFontSize=16, titleFontSize=18, labelLimit=500)),
                strokeDash=alt.StrokeDash("Metric:N", title="Metric",
                                          legend=alt.Legend(labelFontSize=16, titleFontSize=18, labelLimit=300)),
                tooltip=[
                    alt.Tooltip("Scenario:N"),
                    alt.Tooltip("Metric:N"),
                    alt.Tooltip("Value:Q", format=",.3f"),
                    alt.Tooltip("X:N", title="Period"),
                    alt.Tooltip("Location:N"),
                    alt.Tooltip("Name:N"),
                ],
                opacity=alt.condition(sel, alt.value(1), alt.value(0.25)),
            )
            .add_params(sel)
            .properties(height=500, width=1500, background=chart_bg)
        )
        st.altair_chart(chart, use_container_width=False)

    # Baseline preview row
    if use_baseline and not pre_transform_view.empty:
        baseline_slice = pre_transform_view[pre_transform_view["Year"] == y0].copy()
        if len(scen_sel) > 1:
            baseline_table = baseline_slice.pivot_table(
                index=idx_cols, columns=["Data Type", "Scenario"], values="Value", aggfunc="first"
            ).sort_index()
        else:
            baseline_table = baseline_slice.pivot_table(
                index=idx_cols, columns="Data Type", values="Value", aggfunc="first"
            ).sort_index()
        st.subheader(f"Baseline values (Year = {y0})")
        st.dataframe(baseline_table, use_container_width=True)

    # Table
    table_view = view.copy()
    if table_interval > 1 and not table_view.empty and "Year" in table_view.columns:
        anchor = int(table_view["Year"].min())
        table_view = table_view[((table_view["Year"].astype(int) - anchor) % table_interval) == 0]

    if len(scen_sel) > 1:
        table = table_view.pivot_table(index=idx_cols, columns=["Data Type", "Scenario"], values="Value", aggfunc="first").sort_index()
    else:
        table = table_view.pivot_table(index=idx_cols, columns="Data Type", values="Value", aggfunc="first").sort_index()

    st.subheader("Values")
    st.dataframe(table, use_container_width=True, height=520)


# ================================== MAIN ======================================
if __name__ == "__main__":
    if MODE.lower() == "metrics":
        run_metrics_viewer()
    else:
        sys.exit('Set MODE = "metrics"')
