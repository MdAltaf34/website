from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pipeline.forecast_pipeline import (
    FINAL_FORECAST_END,
    FINAL_FORECAST_START,
    SELECTION_END,
    SELECTION_START,
    PipelinePaths,
    TARGET_ASSETS,
    TICKER_ALIASES,
    find_asset_files,
    find_column,
    load_price_data,
    run_pipeline,
)

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data" / "app_data"
OUTPUT_DIR = APP_DIR / "pipeline_output"
RUNTIME_DIR = APP_DIR / ".runtime_inputs"
REPO_INPUT_DIR = APP_DIR / "input_data"
SUMMARY_FILE = DATA_DIR / "companies_summary.csv"
MANIFEST_FILE = DATA_DIR / "run_manifest.json"
FORECAST_COLUMNS = ["date", "Predicted", "Lower", "Upper"]
SENTIMENT_REQUIRED_COLUMNS = {"date", "title_score_mean", "content_score_mean"}

st.set_page_config(
    page_title="DSE One-Month Forecast",
    page_icon="📈",
    layout="wide",
)


def _secret(name: str, default: str = "") -> str:
    """Read a Streamlit secret first, then an environment variable."""
    try:
        value = st.secrets.get(name, default)
        return str(value) if value is not None else default
    except Exception:
        return str(os.getenv(name, default))


def _uploaded_sha256(uploaded_file) -> str:
    return hashlib.sha256(uploaded_file.getvalue()).hexdigest()


def _clear_runtime_inputs() -> None:
    if RUNTIME_DIR.exists():
        shutil.rmtree(RUNTIME_DIR, ignore_errors=True)
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    for key in [
        "validated_upload_hash",
        "validated_input_root",
        "validation_rows",
        "validated_paths",
    ]:
        st.session_state.pop(key, None)


def _safe_extract_zip(uploaded_file) -> Path:
    """Extract an uploaded ZIP into the Streamlit server's temporary runtime area."""
    _clear_runtime_inputs()
    extract_dir = Path(tempfile.mkdtemp(prefix="bundle_", dir=RUNTIME_DIR))
    zip_path = extract_dir / "input_bundle.zip"
    zip_path.write_bytes(uploaded_file.getvalue())

    with zipfile.ZipFile(zip_path) as archive:
        root = extract_dir.resolve()
        for member in archive.infolist():
            # Prevent ../ path traversal and absolute-path extraction.
            target = (extract_dir / member.filename).resolve()
            if root != target and root not in target.parents:
                raise ValueError("The uploaded ZIP contains an unsafe path.")
        archive.extractall(extract_dir)

    zip_path.unlink(missing_ok=True)
    return extract_dir


def _find_exact_file(root: Path, filename: str) -> Path:
    matches = [p for p in root.rglob(filename) if p.is_file()]
    if not matches:
        raise FileNotFoundError(f"{filename} was not found in the uploaded ZIP.")
    if len(matches) > 1:
        raise ValueError(f"More than one {filename} was found. Keep only one copy in the ZIP.")
    return matches[0]


def _asset_file_present(directory: Path, asset: str) -> bool:
    aliases = {x.upper() for x in TICKER_ALIASES[asset]}
    for path in directory.glob("*_data.csv"):
        ticker = path.name.replace("_data.csv", "").upper()
        if ticker in aliases:
            return True
    return False


def _find_price_directory(root: Path) -> Path:
    candidate_dirs = {p.parent for p in root.rglob("*_data.csv") if p.is_file()}
    ranked: list[tuple[int, Path]] = []
    for directory in candidate_dirs:
        count = sum(_asset_file_present(directory, asset) for asset in TARGET_ASSETS)
        ranked.append((count, directory))
    ranked.sort(key=lambda item: item[0], reverse=True)

    if not ranked or ranked[0][0] < len(TARGET_ASSETS):
        found = ranked[0][0] if ranked else 0
        raise FileNotFoundError(
            f"Could not find one folder containing all {len(TARGET_ASSETS)} selected asset files. "
            f"The best folder contained {found}."
        )
    return ranked[0][1]


def _paths_from_root(root: Path) -> PipelinePaths:
    return PipelinePaths(
        price_directory=_find_price_directory(root),
        banglabert_file=_find_exact_file(root, "daily_sentiment_banglabert.csv"),
        lexicon_file=_find_exact_file(root, "daily_sentiment_lexicon.csv"),
        output_directory=OUTPUT_DIR,
        app_data_directory=DATA_DIR,
    )


def _repo_paths() -> PipelinePaths:
    return PipelinePaths(
        price_directory=REPO_INPUT_DIR / "dse-unadjusted-data",
        banglabert_file=REPO_INPUT_DIR / "daily_sentiment_banglabert.csv",
        lexicon_file=REPO_INPUT_DIR / "daily_sentiment_lexicon.csv",
        output_directory=OUTPUT_DIR,
        app_data_directory=DATA_DIR,
    )


def _repo_inputs_ready() -> bool:
    paths = _repo_paths()
    return (
        paths.price_directory.exists()
        and paths.banglabert_file.exists()
        and paths.lexicon_file.exists()
        and all(_asset_file_present(paths.price_directory, a) for a in TARGET_ASSETS)
    )


def _validate_sentiment_file(path: Path, label: str) -> dict:
    data = pd.read_csv(path)
    missing = SENTIMENT_REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")

    dates = pd.to_datetime(data["date"], errors="coerce").dropna()
    if dates.empty:
        raise ValueError(f"{label} has no valid dates.")

    title = pd.to_numeric(data["title_score_mean"], errors="coerce")
    content = pd.to_numeric(data["content_score_mean"], errors="coerce")
    if title.notna().sum() == 0 or content.notna().sum() == 0:
        raise ValueError(f"{label} must contain numeric title and content scores.")

    return {
        "Item": label,
        "Status": "Valid",
        "Rows": len(data),
        "Start": dates.min().date().isoformat(),
        "End": dates.max().date().isoformat(),
        "Note": "Title/content sentiment columns found",
    }


def _validate_market_files(paths: PipelinePaths) -> list[dict]:
    companies = find_asset_files(paths.price_directory)
    rows: list[dict] = []

    for asset in TARGET_ASSETS:
        raw = pd.read_csv(companies[asset])
        date_col = find_column(raw.columns, ["Date", "Trading Date", "Trade Date"])
        close_col = find_column(raw.columns, ["Close", "Closing Price", "Close Price"])
        volume_col = find_column(
            raw.columns,
            ["Volume", "Trade Volume", "Total Volume", "Trading Volume", "Vol", "Vol."],
        )
        if date_col is None or close_col is None:
            raise ValueError(f"{asset}: Date and Close columns are required.")
        if volume_col is None:
            raise ValueError(f"{asset}: Volume is required because VAR is one of the models.")

        cleaned = load_price_data(companies[asset])
        if cleaned.empty:
            raise ValueError(f"{asset}: no usable market rows remain after cleaning.")

        january_rows = cleaned[
            (cleaned["ds"] >= SELECTION_START) & (cleaned["ds"] <= SELECTION_END)
        ]
        pre_january_rows = cleaned[cleaned["ds"] < SELECTION_START]
        if len(pre_january_rows) < 60:
            raise ValueError(f"{asset}: fewer than 60 usable pre-January training rows.")
        if january_rows.empty:
            raise ValueError(f"{asset}: January 2022 data is required for model selection.")

        rows.append(
            {
                "Item": asset,
                "Status": "Valid",
                "Rows": len(cleaned),
                "Start": cleaned["ds"].min().date().isoformat(),
                "End": cleaned["ds"].max().date().isoformat(),
                "Note": f"January selection rows: {len(january_rows)}",
            }
        )
    return rows


def validate_input_package(paths: PipelinePaths) -> pd.DataFrame:
    """Perform strict pre-run checks without training forecasting models."""
    rows = _validate_market_files(paths)
    rows.append(_validate_sentiment_file(paths.banglabert_file, "BanglaBERT sentiment"))
    rows.append(_validate_sentiment_file(paths.lexicon_file, "Lexicon sentiment"))
    return pd.DataFrame(rows)


def _zip_directory_bytes(directory: Path) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        if directory.exists():
            for path in sorted(directory.rglob("*")):
                if path.is_file():
                    archive.write(path, arcname=path.relative_to(directory))
    return buffer.getvalue()


@st.cache_data(show_spinner=False)
def load_summary() -> pd.DataFrame:
    data = pd.read_csv(SUMMARY_FILE)
    required = {
        "Company",
        "Asset_Type",
        "Best_Model",
        "Best_Experiment",
        "Forecast_Start",
        "Forecast_End",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Summary file is missing columns: {sorted(missing)}")
    return data


@st.cache_data(show_spinner=False)
def load_history(company: str) -> pd.DataFrame:
    data = pd.read_csv(DATA_DIR / f"{company}_history.csv", parse_dates=["date"])
    missing = {"date", "Close"}.difference(data.columns)
    if missing:
        raise ValueError(f"History file is missing columns: {sorted(missing)}")
    data["Close"] = pd.to_numeric(data["Close"], errors="coerce")
    return data.dropna(subset=["date", "Close"]).sort_values("date")


@st.cache_data(show_spinner=False)
def load_forecast(company: str) -> pd.DataFrame:
    data = pd.read_csv(DATA_DIR / f"{company}_forecast.csv", parse_dates=["date"])
    missing = set(FORECAST_COLUMNS).difference(data.columns)
    if missing:
        raise ValueError(f"Forecast file is missing columns: {sorted(missing)}")

    # Never expose an Actual field even if a research file accidentally contains one.
    public_data = data[FORECAST_COLUMNS].copy()
    for column in ["Predicted", "Lower", "Upper"]:
        public_data[column] = pd.to_numeric(public_data[column], errors="coerce")
    public_data = public_data.dropna(subset=FORECAST_COLUMNS).sort_values("date")

    if public_data.empty:
        raise ValueError("The forecast contains no valid rows.")
    if (public_data["Lower"] > public_data["Upper"]).any():
        raise ValueError("A lower confidence bound is above its upper bound.")
    return public_data


@st.cache_data(show_spinner=False)
def load_manifest() -> dict:
    if not MANIFEST_FILE.exists():
        return {}
    return json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))


def filter_history(data: pd.DataFrame, choice: str) -> pd.DataFrame:
    if choice == "Full history":
        return data
    days_map = {
        "Last 3 months": 90,
        "Last 6 months": 180,
        "Last 1 year": 365,
    }
    cutoff = data["date"].max() - pd.Timedelta(days=days_map[choice])
    return data[data["date"] >= cutoff]


def run_cloud_forecast(paths: PipelinePaths) -> tuple[pd.DataFrame, dict]:
    progress = st.progress(0, text="Preparing cloud computation...")
    status_box = st.empty()

    def on_progress(done: int, total: int, asset: str, state: str):
        fraction = min(max(done / max(total, 1), 0.0), 1.0)
        progress.progress(
            fraction,
            text=f"{done}/{total} assets processed — {asset}: {state}",
        )
        status_box.caption(
            "January 2022 selects the best model/configuration; February 2022 is forecast blindly."
        )

    summary, manifest = run_pipeline(paths=paths, progress_callback=on_progress)
    progress.progress(1.0, text="Forecast computation and publication completed.")
    return summary, manifest


def admin_login() -> bool:
    configured_password = _secret("ADMIN_PASSWORD", "")
    if not configured_password:
        st.error(
            "Administrator access is disabled because ADMIN_PASSWORD is not configured. "
            "Add it in Streamlit Community Cloud → App settings → Secrets."
        )
        st.code('ADMIN_PASSWORD = "your-private-password"', language="toml")
        return False

    if st.session_state.get("admin_authenticated", False):
        c1, c2 = st.columns([4, 1])
        with c1:
            st.success("Administrator authenticated.")
        with c2:
            if st.button("Log out", use_container_width=True):
                st.session_state["admin_authenticated"] = False
                st.rerun()
        return True

    with st.form("admin_login_form"):
        password = st.text_input("Administrator password", type="password")
        submitted = st.form_submit_button("Log in", type="primary", use_container_width=True)
    if submitted:
        if password == configured_password:
            st.session_state["admin_authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect administrator password.")
    return False


def render_admin_panel() -> None:
    st.title("Administrator Panel")
    st.caption("Private administrator route: /admin")
    st.caption(
        "Upload the research input package, validate it, then run January model selection and "
        "publish the blind February forecast. On Streamlit Community Cloud, all forecasting "
        "computation runs on the cloud app server."
    )

    if not admin_login():
        return

    # Current publication status
    st.subheader("1. Current published forecast")
    if MANIFEST_FILE.exists() and SUMMARY_FILE.exists():
        try:
            manifest = load_manifest()
            summary = load_summary()
            c1, c2, c3 = st.columns(3)
            c1.metric("Status", str(manifest.get("status", "completed")))
            c2.metric("Assets", int(manifest.get("asset_count", len(summary))))
            c3.metric("Published run", str(manifest.get("run_id", "not recorded")))
            st.caption(f"Generated at UTC: {manifest.get('generated_at_utc', 'not recorded')}")
            st.download_button(
                "Download current published app_data.zip",
                data=_zip_directory_bytes(DATA_DIR),
                file_name="app_data_published.zip",
                mime="application/zip",
            )
        except Exception as exc:
            st.warning(f"A published package exists but could not be read: {exc}")
    else:
        st.info("No forecast package is currently published.")

    st.divider()
    st.subheader("2. Upload input data")
    st.write(
        "Use one ZIP containing the eight market CSV files and the two daily sentiment CSV files."
    )
    uploaded_bundle = st.file_uploader(
        "Input data ZIP",
        type=["zip"],
        accept_multiple_files=False,
        help=(
            "Required: DSEX, DS30, DSES, GP, ACI, BEXIMCO, BRACBANK, BXPHARMA market files, "
            "daily_sentiment_banglabert.csv, and daily_sentiment_lexicon.csv."
        ),
    )

    with st.expander("Required ZIP structure"):
        st.code(
            """input_bundle.zip
├── daily_sentiment_banglabert.csv
├── daily_sentiment_lexicon.csv
└── dse-unadjusted-data/
    ├── 00DSEX_data.csv
    ├── 00DS30_data.csv
    ├── 00DSES_data.csv
    ├── GP_data.csv
    ├── ACI_data.csv
    ├── BEXIMCO_data.csv
    ├── BRACBANK_data.csv
    └── BXPHARMA_data.csv""",
            language="text",
        )

    source_paths: PipelinePaths | None = None
    source_label = "Uploaded ZIP"

    # Optional repository mode is convenient while developing the demo.
    if _repo_inputs_ready():
        source_mode = st.radio(
            "Input source",
            ["Uploaded ZIP", "Repository input_data"],
            horizontal=True,
        )
        source_label = source_mode
        if source_mode == "Repository input_data":
            source_paths = _repo_paths()
    else:
        source_mode = "Uploaded ZIP"

    if uploaded_bundle is not None and source_mode == "Uploaded ZIP":
        current_hash = _uploaded_sha256(uploaded_bundle)
        if st.session_state.get("validated_upload_hash") not in (None, current_hash):
            # A different ZIP was selected after an earlier validation.
            for key in ["validated_upload_hash", "validated_input_root", "validation_rows", "validated_paths"]:
                st.session_state.pop(key, None)

        st.caption(
            f"Uploaded: {uploaded_bundle.name} · {uploaded_bundle.size / (1024 * 1024):.1f} MB"
        )

        if st.button("Validate uploaded ZIP", use_container_width=True):
            try:
                with st.spinner("Checking file structure, columns, dates, and January selection data..."):
                    root = _safe_extract_zip(uploaded_bundle)
                    paths = _paths_from_root(root)
                    validation = validate_input_package(paths)
                st.session_state["validated_upload_hash"] = current_hash
                st.session_state["validated_input_root"] = str(root)
                st.session_state["validated_paths"] = paths
                st.session_state["validation_rows"] = validation.to_dict("records")
                st.success("Input package is valid and ready for cloud computation.")
            except Exception as exc:
                st.error(f"Input validation failed: {exc}")

        if st.session_state.get("validated_upload_hash") == current_hash:
            source_paths = st.session_state.get("validated_paths")

    elif source_mode == "Repository input_data":
        if st.button("Validate repository input_data", use_container_width=True):
            try:
                validation = validate_input_package(source_paths)
                st.session_state["validation_rows"] = validation.to_dict("records")
                st.session_state["validated_paths"] = source_paths
                st.success("Repository input data is valid.")
            except Exception as exc:
                st.error(f"Input validation failed: {exc}")
        if st.session_state.get("validated_paths") == source_paths:
            source_paths = st.session_state.get("validated_paths")
        else:
            source_paths = None

    validation_rows = st.session_state.get("validation_rows")
    if validation_rows:
        validation_df = pd.DataFrame(validation_rows)
        st.dataframe(validation_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("3. Run model selection and publish February forecast")
    st.info(
        "January 2022 is used to compare all 3 models × 7 configurations for each asset. "
        "The winning combination is retrained using data available through January. "
        "February Close, Volume, and sentiment are not used in the final forecast."
    )

    ready = source_paths is not None and bool(validation_rows)
    run_clicked = st.button(
        "Run cloud computation & publish February forecast",
        type="primary",
        use_container_width=True,
        disabled=not ready,
    )

    if not ready:
        st.caption("Validate an input package first. The run button is intentionally disabled until validation passes.")

    if run_clicked:
        try:
            with st.spinner(
                "Running Prophet, SARIMAX, and VAR. Keep this browser tab open until the run completes..."
            ):
                summary, manifest = run_cloud_forecast(source_paths)
            st.cache_data.clear()
            st.session_state["last_run_id"] = manifest.get("run_id", "")
            st.success(
                f"Forecast published successfully for {len(summary)} assets. "
                f"Run ID: {manifest.get('run_id', 'not recorded')}"
            )
            st.download_button(
                "Download newly generated app_data.zip",
                data=_zip_directory_bytes(DATA_DIR),
                file_name=f"app_data_{manifest.get('run_id', 'latest')}.zip",
                mime="application/zip",
            )
            st.dataframe(
                summary[
                    [
                        "Company",
                        "Best_Model",
                        "Best_Experiment",
                        "Selection_RMSE",
                        "Forecast_Start",
                        "Forecast_End",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )
        except Exception as exc:
            st.error(f"Forecast computation failed: {exc}")

    st.divider()
    st.caption(
        "Streamlit Community Cloud uses ephemeral local storage. If the free app restarts, uploaded runtime files "
        "and generated app_data may be lost. For a thesis demonstration, download app_data.zip after a successful "
        "run or keep a pre-generated package in the repository as a fallback."
    )


def render_forecast_page() -> None:
    st.sidebar.subheader("Forecast controls")

    if not SUMMARY_FILE.exists():
        st.title("DSE One-Month Stock-Price Forecast")
        st.info(
            "No forecast package is currently published. The Administrator can open /admin, upload the input ZIP, "
            "validate it, and run the cloud computation."
        )
        st.markdown(
            "**Experiment design:** January 2022 is the model-selection month. The selected model/configuration "
            "is retrained through January and used for the blind February 2022 forecast."
        )
        return

    try:
        summary_df = load_summary()
        manifest = load_manifest()
    except Exception as exc:
        st.error(f"The published forecast package could not be validated: {exc}")
        return

    company = st.sidebar.selectbox(
        "Company or index",
        summary_df["Company"].astype(str).tolist(),
    )
    history_range = st.sidebar.radio(
        "Historical context",
        ["Last 3 months", "Last 6 months", "Last 1 year", "Full history"],
        index=0,
    )
    row = summary_df.loc[summary_df["Company"].astype(str) == company].iloc[0]

    try:
        full_history = load_history(company)
        full_forecast = load_forecast(company)
    except Exception as exc:
        st.error(f"The selected asset cannot be displayed: {exc}")
        return

    source_cutoff = full_history["date"].max()
    forecast = full_forecast[full_forecast["date"] > source_cutoff].copy()
    history = filter_history(full_history, history_range)

    if forecast.empty:
        st.error("No future forecast rows exist after the source-data cut-off date.")
        return

    st.sidebar.markdown("---")
    st.sidebar.caption("Forecast information")
    st.sidebar.write(f"**Asset type:** {row['Asset_Type']}")
    st.sidebar.write(f"**Selected model:** {row['Best_Model']}")
    st.sidebar.write(f"**Sentiment setting:** {row['Best_Experiment']}")
    st.sidebar.write(f"**Data cut-off:** {source_cutoff.date()}")
    st.sidebar.write(f"**Forecast period:** {row['Forecast_Start']} to {row['Forecast_End']}")
    selection_rmse = row.get("Selection_RMSE", row.get("RMSE", None))
    if selection_rmse is not None and pd.notna(selection_rmse):
        st.sidebar.write(f"**January selection RMSE:** {float(selection_rmse):.4f}")

    st.title("DSE One-Month Stock-Price Forecast")
    st.subheader(company)

    status = manifest.get("status", "completed")
    generated_at = manifest.get("generated_at_utc", "not recorded")
    run_id = manifest.get("run_id", "not recorded")
    st.success(f"Forecast status: {status} · Run: {run_id}")
    st.caption(f"Generated at (UTC): {generated_at}")
    st.info(
        "January 2022 is used for model/configuration selection. The chart below shows the final blind "
        "February 2022 forecast. February actual prices, Volume, and sentiment are not used in final fitting."
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history["date"],
            y=history["Close"],
            mode="lines",
            name="Historical close",
            line=dict(color="#164e63", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=pd.concat([forecast["date"], forecast["date"][::-1]]),
            y=pd.concat([forecast["Upper"], forecast["Lower"][::-1]]),
            fill="toself",
            fillcolor="rgba(234, 88, 12, 0.16)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name="95% forecast interval",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast["date"],
            y=forecast["Predicted"],
            mode="lines+markers",
            name="Forecasted close",
            line=dict(color="#ea580c", width=2.5),
            marker=dict(size=4),
        )
    )

    cutoff_string = source_cutoff.strftime("%Y-%m-%d")
    fig.add_shape(
        type="line",
        x0=cutoff_string,
        x1=cutoff_string,
        y0=0,
        y1=1,
        xref="x",
        yref="paper",
        line=dict(color="#64748b", width=1.5, dash="dot"),
    )
    fig.add_annotation(
        x=cutoff_string,
        y=1,
        xref="x",
        yref="paper",
        text="Data cut-off",
        showarrow=False,
        yanchor="bottom",
        font=dict(color="#475569", size=11),
    )
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Closing price",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=570,
        margin=dict(l=30, r=20, t=30, b=30),
    )
    st.plotly_chart(fig, use_container_width=True)

    display_table = forecast.rename(
        columns={
            "date": "Date",
            "Predicted": "Forecasted Close",
            "Lower": "Lower Bound",
            "Upper": "Upper Bound",
        }
    )
    with st.expander("View forecast-only table"):
        st.dataframe(display_table, use_container_width=True, hide_index=True)
        st.download_button(
            "Download forecast CSV",
            data=display_table.to_csv(index=False).encode("utf-8"),
            file_name=f"{company}_February_2022_forecast.csv",
            mime="text/csv",
        )

    st.caption(
        "Research prototype only. Forecasts do not guarantee future prices and are not investment advice."
    )


# -----------------------------------------------------------------------------
# URL routing
# -----------------------------------------------------------------------------
# Public page:
#   https://<your-app>.streamlit.app/
#
# Administrator page (not shown in normal navigation):
#   https://<your-app>.streamlit.app/admin
#
# The admin URL is intentionally hidden from the public UI, but it is still
# protected by ADMIN_PASSWORD. Hiding a route is not a security mechanism.
public_page = st.Page(
    render_forecast_page,
    title="DSE Forecast",
    icon=":material/show_chart:",
    default=True,
)

admin_page = st.Page(
    render_admin_panel,
    title="Administrator",
    icon=":material/admin_panel_settings:",
    url_path="admin",
    visibility="hidden",
)

router = st.navigation(
    [public_page, admin_page],
    position="hidden",
)
router.run()
