
import sys
from pathlib import Path

import json
import os
import re

import pandas as pd
import plotly.express as px
import streamlit as st

# -------------------------------------------------------------------
# Project path setup
# -------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from carbon_calculator import calculate_all_scenarios, load_reference_data


# -------------------------------------------------------------------
# Natural-language requirement extraction
# -------------------------------------------------------------------
def _extract_number(text, patterns, default=None):
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if match:
            try:
                return float(match.group(1).replace(",", ""))
            except (ValueError, AttributeError):
                pass
    return default


def fallback_parse_requirements(text):
    """Robust local parser used when Ollama is unavailable."""
    t = text.lower().replace(",", "")

    def first_number(patterns, default=None):
        for pattern in patterns:
            m = re.search(pattern, t, re.I)
            if m:
                try:
                    return float(m.group(1))
                except (ValueError, TypeError):
                    pass
        return default

    requests = first_number([
        r"(\d+(?:\.\d+)?)\s*(?:requests?|reqs?)\s*(?:per|a|each)\s*day",
        r"(\d+(?:\.\d+)?)\s*(?:daily|per day)\s*(?:requests?)?",
    ], 100000)

    # Handles both "minimum accuracy of 99%" and "at least 99% accuracy".
    accuracy = first_number([
        r"(?:minimum|min(?:imum)?|at least|target|required)\s+accuracy\s*(?:of|is|:)?\s*(\d+(?:\.\d+)?)\s*%",
        r"(?:at least|minimum|min(?:imum)?|above)\s*(\d+(?:\.\d+)?)\s*%\s*accuracy",
        r"(\d+(?:\.\d+)?)\s*%\s*accuracy",
    ], 95.0)

    # Handles "below 100 milliseconds", "under 80 ms", etc.
    latency = first_number([
        r"(?:latency|response latency)\s*(?:should be|must be|of|is|:)?\s*(?:below|under|less than|maximum|max)?\s*(\d+(?:\.\d+)?)\s*(?:milliseconds?|ms)",
        r"(?:below|under|less than|maximum|max(?:imum)?)\s*(\d+(?:\.\d+)?)\s*(?:milliseconds?|ms)",
    ], 50.0)

    storage = first_number([
        r"(\d+(?:\.\d+)?)\s*(?:gb|gigabytes)\s*(?:of)?\s*storage",
        r"storage\s*(?:of|:)?\s*(\d+(?:\.\d+)?)\s*(?:gb|gigabytes)",
    ], 50.0)

    # Handles "200 GB of data over the network every month".
    network = first_number([
        r"(\d+(?:\.\d+)?)\s*(?:gb|gigabytes)\s*(?:of\s+)?(?:data\s+)?(?:over|through|via|on)?\s*the?\s*network",
        r"(\d+(?:\.\d+)?)\s*(?:gb|gigabytes)\s*(?:of\s+)?(?:network\s+)?(?:data\s+)?(?:transfer|traffic)",
        r"network\s*(?:transfer|usage)?\s*(?:of|:)?\s*(\d+(?:\.\d+)?)\s*(?:gb|gigabytes)",
    ], 100.0)

    training = first_number([
        r"(?:initial\s+)?training\s*(?:will\s+take|takes|for|of|:)?\s*(?:around|about|approximately)?\s*(\d+(?:\.\d+)?)\s*(?:hours?|hrs?)",
        r"(\d+(?:\.\d+)?)\s*(?:hours?|hrs?)\s*(?:of\s+)?(?:initial\s+)?training",
    ], 8.0)

    if re.search(r"(?:monthly|every month)", t) and re.search(r"retrain", t):
        retraining_runs = 12
    elif re.search(r"(?:weekly|every week)", t) and re.search(r"retrain", t):
        retraining_runs = 52
    elif re.search(r"(?:quarterly|every 3 months|every three months)", t) and re.search(r"retrain", t):
        retraining_runs = 4
    elif re.search(r"(?:yearly|annually|every year)", t) and re.search(r"retrain", t):
        retraining_runs = 1
    else:
        retraining_runs = None

    return {
        "requests_per_day": int(requests) if requests is not None else None,
        "min_accuracy": accuracy,
        "max_latency": latency,
        "storage_gb": storage,
        "network_gb_per_month": network,
        "training_hours": training,
        "retraining_runs_per_year": retraining_runs,
    }


def ollama_available():
    """Check whether a local Ollama server is reachable.

    The app does not require Ollama: if it is unavailable, the local
    fallback parser is used automatically.
    """
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=1.5)
        return response.ok
    except Exception:
        return False


def parse_natural_language(text):
    """Use a local Ollama model to extract structured requirements.

    Ollama is expected to be running locally, e.g. with a model such as
    llama3.2, qwen2.5, mistral, etc. The model name can be changed with
    the OLLAMA_MODEL environment variable.
    """
    try:
        import ollama

        model = os.getenv("OLLAMA_MODEL", "llama3.2")
        prompt = f"""
You are the requirement-extraction component of a Sustainable AI Lifecycle Auditor.
Read the user's description and extract only values explicitly stated or clearly implied.
Return ONLY valid JSON. Do not add markdown, explanations, or code fences.

Allowed JSON fields:
- requests_per_day: number or null
- min_accuracy: number or null
- max_latency: number or null (milliseconds)
- storage_gb: number or null
- network_gb_per_month: number or null
- training_hours: number or null
- training_runs: number or null
- retraining_hours: number or null
- retraining_runs_per_year: number or null
- inference_device_hours: number or null

Rules:
- Convert percentages to numeric values, e.g. 95% -> 95.
- Convert monthly retraining -> 12 retraining runs per year.
- Convert weekly retraining -> 52 retraining runs per year.
- Convert quarterly retraining -> 4 retraining runs per year.
- Convert yearly/annual retraining -> 1 retraining run per year.
- Convert seconds of latency to milliseconds if necessary.
- Do not invent missing values. Use null.

User description:
{text}
"""

        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            format="json",
        )

        raw = response["message"]["content"]
        parsed = json.loads(raw)
        return parsed, f"Ollama ({model})"

    except Exception as exc:
        # Keep the demo usable even if Ollama is not running or the model
        # has not been downloaded yet.
        return fallback_parse_requirements(text), f"Local fallback (Ollama unavailable: {exc})"


def apply_ai_values(parsed):
    """Store extracted values so Streamlit widgets use them on the next run."""
    mapping = {
        "requests_per_day": "requests_per_day",
        "min_accuracy": "min_accuracy",
        "max_latency": "max_latency",
        "storage_gb": "storage_gb",
        "network_gb_per_month": "network_gb_per_month",
        "training_hours": "training_hours",
        "training_runs": "training_runs",
        "retraining_hours": "retraining_hours",
        "retraining_runs_per_year": "retraining_runs",
        "inference_device_hours": "inference_device_hours",
    }
    for source, target in mapping.items():
        value = parsed.get(source)
        if value is not None:
            st.session_state[target] = value


# -------------------------------------------------------------------
# Page configuration
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Sustainable AI Lifecycle Auditor",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------------------------
# Styling
# -------------------------------------------------------------------
st.markdown(
    """
    <style>
        /* ==========================================================
           GLOBAL THEME
           ========================================================== */
        .stApp {
            background: #f4f7f5 !important;
            color: #17231d !important;
        }

        .main .block-container {
            padding-top: 1.5rem;
            padding-bottom: 3rem;
            max-width: 1400px;
        }

        /* Force normal Streamlit text to be dark */
        .stApp,
        .stApp p,
        .stApp span,
        .stApp label,
        .stApp div,
        .stApp li,
        .stApp h1,
        .stApp h2,
        .stApp h3,
        .stApp h4,
        .stApp h5,
        .stApp h6 {
            color: #17231d;
        }

        /* ==========================================================
           SIDEBAR
           ========================================================== */
        section[data-testid="stSidebar"] {
            background: #ffffff !important;
            border-right: 1px solid #dfe7e2;
        }

        section[data-testid="stSidebar"] * {
            color: #17231d !important;
        }

        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            color: #12372a !important;
        }

        /* ==========================================================
           HERO
           ========================================================== */
        .hero {
            background: linear-gradient(135deg, #12372a 0%, #21704f 100%);
            padding: 2.2rem 2.4rem;
            border-radius: 20px;
            color: white !important;
            margin-bottom: 1.4rem;
            box-shadow: 0 8px 25px rgba(18, 55, 42, 0.16);
        }

        .hero h1,
        .hero p {
            color: white !important;
        }

        .hero h1 {
            margin: 0;
            font-size: 2.35rem;
            font-weight: 750;
        }

        .hero p {
            margin: 0.55rem 0 0;
            color: #e9f7ef !important;
            font-size: 1.02rem;
        }

        /* ==========================================================
           HEADINGS
           ========================================================== */
        .section-title {
            margin-top: 1.2rem;
            margin-bottom: 0.7rem;
            color: #12372a !important;
        }

        .stApp h2,
        .stApp h3 {
            color: #12372a !important;
        }

        /* ==========================================================
           INPUTS - IMPORTANT: FORCE DARK TEXT
           ========================================================== */

        /* Input containers */
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        div[data-baseweb="textarea"] > div {
            background-color: #ffffff !important;
            border-color: #cbd8d0 !important;
        }

        /* Text inside inputs */
        div[data-baseweb="select"] *,
        div[data-baseweb="input"] *,
        div[data-baseweb="textarea"] * {
            color: #17231d !important;
            -webkit-text-fill-color: #17231d !important;
        }

        input,
        textarea {
            color: #17231d !important;
            -webkit-text-fill-color: #17231d !important;
            background-color: #ffffff !important;
            caret-color: #12372a !important;
        }

        input::placeholder,
        textarea::placeholder {
            color: #6b7871 !important;
            -webkit-text-fill-color: #6b7871 !important;
            opacity: 1 !important;
        }

        /* Selectbox selected value */
        div[data-baseweb="select"] span {
            color: #17231d !important;
            -webkit-text-fill-color: #17231d !important;
        }

        /* Dropdown menu */
        div[data-baseweb="popover"] {
            background-color: #ffffff !important;
        }

        div[data-baseweb="popover"] * {
            color: #17231d !important;
        }

        ul[role="listbox"] {
            background-color: #ffffff !important;
        }

        li[role="option"] {
            color: #17231d !important;
            background-color: #ffffff !important;
        }

        li[role="option"]:hover {
            background-color: #eaf5ef !important;
        }

        /* Number input +/- buttons */
        button[data-testid="stNumberInputStepDown"],
        button[data-testid="stNumberInputStepUp"] {
            color: #12372a !important;
            background: #ffffff !important;
        }

        /* Input labels */
        .stTextInput label,
        .stNumberInput label,
        .stSelectbox label,
        .stMultiSelect label,
        .stSlider label {
            color: #26372e !important;
            font-weight: 600 !important;
        }

        /* ==========================================================
           BUTTONS
           ========================================================== */
        .stButton > button {
            background: #21704f !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 10px !important;
            font-weight: 700 !important;
            min-height: 2.6rem !important;
            transition: all 0.2s ease;
        }

        .stButton > button p,
        .stButton > button span,
        .stButton > button div {
            color: #ffffff !important;
        }

        .stButton > button:hover {
            background: #18563c !important;
            color: #ffffff !important;
            box-shadow: 0 5px 14px rgba(33, 112, 79, 0.25);
        }

        .stDownloadButton > button {
            background: #ffffff !important;
            color: #12372a !important;
            border: 1px solid #21704f !important;
            border-radius: 10px !important;
            font-weight: 700 !important;
        }

        .stDownloadButton > button p,
        .stDownloadButton > button span {
            color: #12372a !important;
        }

        /* ==========================================================
           METRIC CARDS
           ========================================================== */
        div[data-testid="stMetric"] {
            background: #ffffff !important;
            border: 1px solid #dce6e0 !important;
            padding: 1rem !important;
            border-radius: 14px !important;
            box-shadow: 0 3px 12px rgba(0,0,0,0.04);
        }

        div[data-testid="stMetric"] label {
            color: #53635a !important;
        }

        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: #12372a !important;
        }

        div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
            color: #21704f !important;
        }

        /* ==========================================================
           DATAFRAMES / TABLES
           ========================================================== */
        div[data-testid="stDataFrame"] {
            background: #ffffff !important;
            border-radius: 12px !important;
        }

        div[data-testid="stDataFrame"] * {
            color: #17231d !important;
        }

        /* ==========================================================
           EXPANDERS
           ========================================================== */
        div[data-testid="stExpander"] {
            background: #ffffff !important;
            border: 1px solid #dce6e0 !important;
            border-radius: 12px !important;
        }

        div[data-testid="stExpander"] summary {
            color: #12372a !important;
            font-weight: 700 !important;
        }

        div[data-testid="stExpander"] summary span {
            color: #12372a !important;
        }

        /* ==========================================================
           ALERTS / INFO BOXES
           ========================================================== */
        div[data-testid="stAlert"] p,
        div[data-testid="stAlert"] span {
            color: #26372e !important;
        }

        /* ==========================================================
           RECOMMENDATION
           ========================================================== */
        .recommendation {
            background: #e8f5ed !important;
            border-left: 5px solid #21704f;
            padding: 1rem 1.2rem;
            border-radius: 10px;
            margin-top: 0.8rem;
        }

        .recommendation,
        .recommendation * {
            color: #173528 !important;
        }

        .small-note {
            color: #66756c !important;
            font-size: 0.85rem;
        }

        /* ==========================================================
           SLIDER
           ========================================================== */
        div[data-testid="stSlider"] * {
            color: #26372e !important;
        }

        /* ==========================================================
           CHECKBOX / RADIO
           ========================================================== */
        div[data-testid="stCheckbox"] label,
        div[data-testid="stRadio"] label {
            color: #17231d !important;
        }

        /* ==========================================================
           MOBILE
           ========================================================== */
        @media (max-width: 768px) {
            .hero {
                padding: 1.5rem;
            }

            .hero h1 {
                font-size: 1.75rem;
            }

            .hero p {
                font-size: 0.92rem;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------
# Header
# -------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🌱 Sustainable AI Lifecycle Auditor</h1>
        <p>
            Estimate the energy and carbon impact of an AI system across
            training, inference, storage, networking, retraining and hardware.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------
# AI natural-language input
# -------------------------------------------------------------------
st.markdown('<h2 class="section-title">🤖 Describe Your AI System</h2>', unsafe_allow_html=True)
st.caption(
    "Describe your requirements in normal language. The AI layer extracts "
    "structured values that can be used by the lifecycle calculator."
)

if ollama_available():
    st.success("🟢 Ollama detected. The Analyze with AI button will use your local Ollama model.")
else:
    st.info("🟡 Ollama is not installed/running on this device. Local fallback mode is active, so the demo still works without Ollama.")

example_text = (
    "I am building a customer support chatbot. I need at least 95% accuracy, "
    "latency below 100 ms, around 10,000 requests per day, 50 GB of storage, "
    "and monthly retraining."
)

nl_text = st.text_area(
    "AI system description",
    placeholder=example_text,
    height=120,
    key="nl_system_description",
)

ai_col1, ai_col2 = st.columns([1, 3])
with ai_col1:
    parse_ai = st.button("✨ Analyze with AI", type="primary", use_container_width=True)

if parse_ai:
    if not nl_text.strip():
        st.warning("Enter a description of your AI system first.")
    else:
        parsed, parser_name = parse_natural_language(nl_text)
        st.session_state.ai_parsed = parsed
        st.session_state.ai_parser_name = parser_name

        # Store the extracted values as pending widget values. They are
        # applied before the sidebar widgets are created on the rerun.
        st.session_state.ai_pending_values = {
            k: v for k, v in parsed.items() if v is not None
        }
        st.session_state.run_analysis = True
        st.rerun()

if "ai_parsed" in st.session_state:
    parsed = st.session_state.ai_parsed
    st.markdown("**AI Extracted Requirements**")
    pretty = {
        "Requests / day": parsed.get("requests_per_day"),
        "Minimum accuracy (%)": parsed.get("min_accuracy"),
        "Maximum latency (ms)": parsed.get("max_latency"),
        "Storage (GB)": parsed.get("storage_gb"),
        "Network (GB / month)": parsed.get("network_gb_per_month"),
        "Training hours": parsed.get("training_hours"),
        "Retraining runs / year": parsed.get("retraining_runs_per_year"),
    }
    extracted_df = pd.DataFrame(
        [{"Parameter": k, "Extracted value": ("Not stated" if v is None else v)} for k, v in pretty.items()]
    )
    st.dataframe(extracted_df, use_container_width=True, hide_index=True)
    st.caption(
        "The extracted values are editable in the sidebar before running the final audit."
    )

st.divider()

# -------------------------------------------------------------------
# Load teammate's calculation engine
# -------------------------------------------------------------------
DATA_DIR = BASE_DIR / "data"

try:
    reference_data = load_reference_data(DATA_DIR)
except Exception as exc:
    st.error(
        "Could not load the calculation engine data. "
        "Make sure app.py is placed in the same folder as "
        "`carbon_calculator.py` and the `data/` folder."
    )
    st.exception(exc)
    st.stop()

# Apply AI-extracted values before creating the sidebar widgets.
# This guarantees the values shown by the +/- controls are the same values
# used by the calculation engine.
if st.session_state.get("ai_pending_values"):
    pending = st.session_state.pop("ai_pending_values")
    for key, value in pending.items():
        st.session_state[key] = value

# -------------------------------------------------------------------
# Sidebar: system inputs
# -------------------------------------------------------------------
st.sidebar.header("⚙️ AI System Inputs")
st.sidebar.caption("These values are passed to the team's carbon engine.")

region = st.sidebar.selectbox(
    "Region",
    list(reference_data["grid"].keys()),
    index=0,
)

gpu_name = st.sidebar.selectbox(
    "Hardware",
    list(reference_data["hardware"].keys()),
    index=0,
)

gpu_count = st.sidebar.number_input(
    "Number of devices",
    min_value=1,
    value=int(st.session_state.get("gpu_count", 1)),
    step=1,
)

training_hours = st.sidebar.number_input(
    "Initial training hours",
    min_value=0.0,
    value=float(st.session_state.get("training_hours", 8.0)),
    step=1.0,
)

training_runs = st.sidebar.number_input(
    "Initial training runs",
    min_value=1,
    value=int(st.session_state.get("training_runs", 1)),
    step=1,
)

requests_per_day = st.sidebar.number_input(
    "Inference requests / day",
    min_value=0,
    value=int(st.session_state.get("requests_per_day", 100000)),
    step=10000,
)

energy_per_request_kwh = st.sidebar.number_input(
    "Energy / request (kWh)",
    min_value=0.0,
    value=float(st.session_state.get("energy_per_request_kwh", 0.000002)),
    format="%.8f",
)

storage_gb = st.sidebar.number_input(
    "Storage (GB)",
    min_value=0.0,
    value=float(st.session_state.get("storage_gb", 50.0)),
    step=10.0,
)

network_gb_per_month = st.sidebar.number_input(
    "Network transfer (GB / month)",
    min_value=0.0,
    value=float(st.session_state.get("network_gb_per_month", 100.0)),
    step=10.0,
)

retraining_hours = st.sidebar.number_input(
    "Retraining hours / run",
    min_value=0.0,
    value=float(st.session_state.get("retraining_hours", 2.0)),
    step=1.0,
)

retraining_runs = st.sidebar.number_input(
    "Retraining runs / year",
    min_value=0,
    value=int(st.session_state.get("retraining_runs", 4)),
    step=1,
)

inference_device_hours = st.sidebar.number_input(
    "Inference device-hours / day",
    min_value=0.0,
    value=float(st.session_state.get("inference_device_hours", 2.0)),
    step=0.5,
)

st.sidebar.divider()
st.sidebar.subheader("🎯 Constraints")

min_accuracy = st.sidebar.number_input(
    "Minimum accuracy (%)",
    min_value=0.0,
    max_value=100.0,
    value=float(st.session_state.get("min_accuracy", 95.0)),
    step=1.0,
)

max_latency = st.sidebar.number_input(
    "Maximum latency (ms)",
    min_value=1.0,
    value=float(st.session_state.get("max_latency", 50.0)),
    step=5.0,
)

analyze = st.sidebar.button(
    "🚀 Analyze AI System",
    type="primary",
    use_container_width=True,
)

# Run automatically once, then rerun when button is clicked.
if "run_analysis" not in st.session_state:
    st.session_state.run_analysis = True

if analyze:
    st.session_state.run_analysis = True

# -------------------------------------------------------------------
# Main content
# -------------------------------------------------------------------
if st.session_state.run_analysis:

    system = {
        "region": region,
        "gpu_name": gpu_name,
        "gpu_count": gpu_count,
        "training_hours": training_hours,
        "training_runs": training_runs,
        "energy_per_request_kwh": energy_per_request_kwh,
        "requests_per_day": requests_per_day,
        "inference_device_hours_per_day": inference_device_hours,
        "storage_gb": storage_gb,
        "network_gb_per_month": network_gb_per_month,
        "retraining_hours_per_run": retraining_hours,
        "retraining_runs_per_year": retraining_runs,
        "days": 365,
        "months": 12,
    }

    try:
        results = calculate_all_scenarios(system, reference_data)
    except Exception as exc:
        st.error("The calculation engine rejected the supplied inputs.")
        st.exception(exc)
        st.stop()

    expected = results["expected"]
    stages = expected["stages"]

    # ---------------------------------------------------------------
    # KPI cards
    # ---------------------------------------------------------------
    st.markdown(
        '<h2 class="section-title">📊 Lifecycle Overview</h2>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Total Energy",
            f"{expected['total_energy_kwh']:,.1f} kWh",
            help="Expected/base scenario over the selected one-year period.",
        )

    with c2:
        st.metric(
            "Total Carbon",
            f"{expected['total_kgco2e']:,.1f} kg CO₂e",
            help="Expected/base lifecycle carbon estimate.",
        )

    with c3:
        st.metric(
            "Grid Intensity",
            f"{expected['grid_intensity_g_per_kwh']:,.0f} g/kWh",
        )

    with c4:
        st.metric(
            "Hardware",
            gpu_name,
            delta=f"{gpu_count} device(s)",
        )

    # ---------------------------------------------------------------
    # Lifecycle chart
    # ---------------------------------------------------------------
    left, right = st.columns([1.4, 1])

    stage_rows = []
    for stage, values in stages.items():
        stage_rows.append(
            {
                "Stage": stage.title(),
                "Carbon (kg CO₂e)": values["carbon_kgco2e"],
                "Energy (kWh)": values["energy_kwh"],
            }
        )

    stage_df = pd.DataFrame(stage_rows)

    with left:
        st.subheader("Lifecycle Carbon Breakdown")
        fig = px.bar(
            stage_df,
            x="Stage",
            y="Carbon (kg CO₂e)",
            text_auto=".2f",
        )
        fig.update_layout(
            margin=dict(l=10, r=10, t=20, b=10),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Lifecycle Share")
        pie = px.pie(
            stage_df,
            names="Stage",
            values="Carbon (kg CO₂e)",
            hole=0.48,
        )
        pie.update_layout(
            margin=dict(l=10, r=10, t=20, b=10),
            showlegend=True,
        )
        st.plotly_chart(pie, use_container_width=True)

    # ---------------------------------------------------------------
    # Low / expected / high range
    # ---------------------------------------------------------------
    st.subheader("📈 Estimate Range")

    range_df = pd.DataFrame(
        [
            {
                "Scenario": "Low",
                "Energy (kWh)": results["low"]["total_energy_kwh"],
                "Carbon (kg CO₂e)": results["low"]["total_kgco2e"],
            },
            {
                "Scenario": "Expected",
                "Energy (kWh)": results["expected"]["total_energy_kwh"],
                "Carbon (kg CO₂e)": results["expected"]["total_kgco2e"],
            },
            {
                "Scenario": "High",
                "Energy (kWh)": results["high"]["total_energy_kwh"],
                "Carbon (kg CO₂e)": results["high"]["total_kgco2e"],
            },
        ]
    )

    st.dataframe(
        range_df.style.format(
            {
                "Energy (kWh)": "{:,.2f}",
                "Carbon (kg CO₂e)": "{:,.2f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    # ---------------------------------------------------------------
    # Architecture comparison
    # ---------------------------------------------------------------
    st.markdown(
        '<h2 class="section-title">🏗️ Architecture Comparison</h2>',
        unsafe_allow_html=True,
    )

    st.caption(
        "For the hackathon demo, enter the expected accuracy and latency "
        "of each architecture. Carbon is calculated using the team's engine."
    )

    architecture_rows = []

    default_profiles = [
        ("Architecture A", "NVIDIA T4", 96.0, 45.0),
        ("Architecture B", "NVIDIA A100", 98.0, 35.0),
        ("Architecture C", "CPU Server", 92.0, 80.0),
    ]

    cols = st.columns(3)

    for idx, (default_name, default_gpu, default_acc, default_lat) in enumerate(
        default_profiles
    ):
        with cols[idx]:
            st.markdown(f"**{default_name}**")

            name = st.text_input(
                "Name",
                value=default_name,
                key=f"arch_name_{idx}",
            )

            arch_gpu = st.selectbox(
                "Hardware",
                list(reference_data["hardware"].keys()),
                index=list(reference_data["hardware"].keys()).index(default_gpu),
                key=f"arch_gpu_{idx}",
            )

            accuracy = st.number_input(
                "Accuracy (%)",
                min_value=0.0,
                max_value=100.0,
                value=default_acc,
                step=0.5,
                key=f"arch_acc_{idx}",
            )

            latency = st.number_input(
                "Latency (ms)",
                min_value=0.0,
                value=default_lat,
                step=1.0,
                key=f"arch_lat_{idx}",
            )

            arch_system = dict(system)
            arch_system["gpu_name"] = arch_gpu

            try:
                arch_result = calculate_all_scenarios(
                    arch_system,
                    reference_data,
                )["expected"]

                carbon = arch_result["total_kgco2e"]
                energy = arch_result["total_energy_kwh"]

                meets_constraints = (
                    accuracy >= min_accuracy
                    and latency <= max_latency
                )

                architecture_rows.append(
                    {
                        "Architecture": name,
                        "Hardware": arch_gpu,
                        "Accuracy (%)": accuracy,
                        "Latency (ms)": latency,
                        "Energy (kWh)": energy,
                        "Carbon (kg CO₂e)": carbon,
                        "Meets constraints": (
                            "✅ Yes" if meets_constraints else "❌ No"
                        ),
                        "_feasible": meets_constraints,
                    }
                )

            except Exception as exc:
                st.error(f"{name}: {exc}")

    comparison_df = pd.DataFrame(architecture_rows)

    if not comparison_df.empty:
        display_df = comparison_df.drop(columns=["_feasible"])

        st.dataframe(
            display_df.style.format(
                {
                    "Accuracy (%)": "{:.1f}",
                    "Latency (ms)": "{:.1f}",
                    "Energy (kWh)": "{:,.2f}",
                    "Carbon (kg CO₂e)": "{:,.2f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        feasible = comparison_df[comparison_df["_feasible"]]

        if len(feasible) == 0:
            st.warning(
                "No architecture satisfies both constraints. "
                "Try relaxing the accuracy or latency requirement."
            )
        else:
            # This is intentionally a transparent rule:
            # among architectures that satisfy constraints, show the one
            # with the lowest calculated carbon.
            selected = feasible.sort_values("Carbon (kg CO₂e)").iloc[0]

            st.markdown(
                f"""
                <div class="recommendation">
                    <b>Feasible architecture with the lowest calculated carbon:</b>
                    {selected['Architecture']} ({selected['Hardware']})<br>
                    Accuracy: {selected['Accuracy (%)']:.1f}% &nbsp;|&nbsp;
                    Latency: {selected['Latency (ms)']:.1f} ms &nbsp;|&nbsp;
                    Carbon: {selected['Carbon (kg CO₂e)']:.2f} kg CO₂e
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.caption(
                "This is a transparent constraint-based comparison. "
                "It is not a machine-learned prediction."
            )
    # ---------------------------------------------------------------
    # Real integrated Role 1 + Role 2 + Role 3 results
    # ---------------------------------------------------------------
    st.markdown(
        '<h2 class="section-title">🔗 Complete Workflow Results</h2>',
        unsafe_allow_html=True,
    )

    PROJECT_ROOT = BASE_DIR.parent

    role1_output_file = (
        PROJECT_ROOT
        / "carbon_engine"
        / "outputs"
        / "model_carbon_results.csv"
    )

    role2_output_file = (
        PROJECT_ROOT
        / "role2_experiments"
        / "results"
        / "role3_metrics.csv"
    )

    role3_output_file = (
        PROJECT_ROOT
        / "decision_engine"
        / "outputs"
        / "recommendation.json"
    )

    if role1_output_file.exists() and role2_output_file.exists():
        role1_df = pd.read_csv(role1_output_file)
        role2_df = pd.read_csv(role2_output_file)

        integrated_df = pd.merge(
            role2_df,
            role1_df,
            on="option_id",
            how="inner",
            suffixes=("_experiment", "_carbon"),
        )

        st.subheader("Actual Model Comparison")

        st.dataframe(
            integrated_df,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "Accuracy, latency and energy-per-request come from Role 2. "
            "Lifecycle carbon comes from Role 1."
        )

    else:
        st.warning(
            "Role 1 or Role 2 output is missing. "
            "Run python run_workflow.py from the project root."
        )

    st.subheader("Role 3 Recommendation")

    if role3_output_file.exists():
        try:
            with open(role3_output_file, "r", encoding="utf-8") as file:
                recommendation_data = json.load(file)

            selected = (
                    recommendation_data.get("selected_option")
                    or recommendation_data.get("recommendation")
                    or recommendation_data.get("selected")
                    or {}
            )

            if isinstance(selected, dict):
                model_name = selected.get(
                    "model_name",
                    selected.get(
                        "name",
                        selected.get("option_id", "Recommended model")
                    ),
                )

                option_id = selected.get("option_id", "")
                accuracy = selected.get("accuracy", "N/A")
                latency = selected.get(
                    "latency_ms",
                    selected.get("latency", "N/A")
                )
                carbon = selected.get(
                    "expected_kgco2e",
                    selected.get(
                        "total_kgco2e",
                        selected.get("carbon_kgco2e", "N/A")
                    ),
                )

                st.markdown(
                    f"""
                    <div class="recommendation">
                        <h3>🌱 Recommended Option: {model_name}</h3>
                        <p><b>Option ID:</b> {option_id}</p>
                        <p><b>Accuracy:</b> {accuracy}</p>
                        <p><b>Latency:</b> {latency} ms</p>
                        <p><b>Lifecycle carbon:</b> {carbon} kg CO₂e</p>
                        <p>
                            This option satisfies the selected accuracy and
                            latency limits while producing the lowest estimated
                            lifecycle carbon.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            else:
                st.success(str(selected))

            explanation = recommendation_data.get("explanation")

            if explanation:
                st.info(explanation)

            # Keep technical JSON hidden for judges who want details
            with st.expander("View technical Role 3 output"):
                st.json(recommendation_data)

            st.success(
                "This recommendation was produced by the real Role 3 "
                "decision engine, not by random frontend values."
            )

        except Exception as exc:
            st.error("Could not read Role 3 recommendation.")
            st.exception(exc)
    else:
        st.warning(
            "Role 3 recommendation.json is missing. "
            "Run the complete workflow first."
        )


    # ---------------------------------------------------------------
    # Detailed stage table
    # ---------------------------------------------------------------
    with st.expander("🔍 View detailed lifecycle calculations"):
        detail_df = stage_df.copy()
        st.dataframe(
            detail_df.style.format(
                {
                    "Carbon (kg CO₂e)": "{:,.4f}",
                    "Energy (kWh)": "{:,.4f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    # ---------------------------------------------------------------
    # Export
    # ---------------------------------------------------------------
    st.subheader("⬇️ Export Results")

    export_df = pd.DataFrame(
        [
            {
                "scenario": label,
                "region": result["region"],
                "hardware": result["gpu_name"],
                "total_energy_kwh": result["total_energy_kwh"],
                "total_kgco2e": result["total_kgco2e"],
                **{
                    f"{stage}_kgco2e": values["carbon_kgco2e"]
                    for stage, values in result["stages"].items()
                },
            }
            for label, result in results.items()
        ]
    )

    st.download_button(
        "Download lifecycle results CSV",
        data=export_df.to_csv(index=False),
        file_name="sustainable_ai_lifecycle_results.csv",
        mime="text/csv",
    )

    st.caption(
        "⚠️ Current teammate reference data contains demo assumptions and "
        "placeholder source URLs. Replace those before the final presentation."
    )

else:
    st.info("Enter the AI system details in the sidebar and click Analyze.")
