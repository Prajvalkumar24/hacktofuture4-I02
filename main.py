from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st

CSV_PATH = Path("argo_data.csv")

st.set_page_config(
    page_title="Team Sphinx | Ocean Agent",
    layout="wide",
    page_icon="🌊",
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    .stSidebar {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    .stSidebar [data-testid="stMarkdownContainer"],
    .stSidebar label,
    .stSidebar .stCaption,
    .stSidebar .stSelectbox label {
        color: #c9d1d9;
    }
    .stTextInput > div > div > input,
    .stChatInput textarea,
    .stSelectbox > div > div {
        background-color: #010409;
        color: #f0f6fc;
        border: 1px solid #30363d;
    }
    div[data-testid="stChatMessage"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.75rem;
    }
    div[data-testid="stExpander"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
    }
    .stSpinner > div > div {
        border-top-color: #58a6ff !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lowered = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run converter.py first.")
    return pd.read_csv(path)


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def retrieve_context_rows(df: pd.DataFrame, question: str, k: int = 30) -> pd.DataFrame:
    """Retrieve likely relevant rows using lightweight semantic heuristics."""
    temp_col = _find_column(df, ["temperature", "temp", "temp_adjusted"])
    depth_col = _find_column(df, ["depth", "pres", "pressure"])
    time_col = _find_column(df, ["time", "juld"])
    lat_col = _find_column(df, ["latitude", "lat"])
    lon_col = _find_column(df, ["longitude", "lon"])

    if not temp_col:
        return df.head(k)

    work = df.copy()
    work[temp_col] = pd.to_numeric(work[temp_col], errors="coerce")
    work = work[work[temp_col].notna()]
    if work.empty:
        return df.head(k)

    q = question.lower().strip()

    number_match = re.search(r"-?\d+(\.\d+)?", q)
    query_number = _safe_float(number_match.group(0)) if number_match else None

    if depth_col and "depth" in q and query_number is not None:
        work[depth_col] = pd.to_numeric(work[depth_col], errors="coerce")
        depth_rows = work[work[depth_col].notna()].copy()
        if not depth_rows.empty:
            depth_rows["score"] = (depth_rows[depth_col] - query_number).abs()
            return depth_rows.nsmallest(k, "score")

    if lat_col and lon_col and query_number is not None and any(
        word in q for word in ["lat", "lon", "longitude", "latitude", "near"]
    ):
        work[lat_col] = pd.to_numeric(work[lat_col], errors="coerce")
        work[lon_col] = pd.to_numeric(work[lon_col], errors="coerce")
        if "lat" in q:
            lat_rows = work[work[lat_col].notna()].copy()
            if not lat_rows.empty:
                lat_rows["score"] = (lat_rows[lat_col] - query_number).abs()
                return lat_rows.nsmallest(k, "score")
        if "lon" in q or "long" in q:
            lon_rows = work[work[lon_col].notna()].copy()
            if not lon_rows.empty:
                lon_rows["score"] = (lon_rows[lon_col] - query_number).abs()
                return lon_rows.nsmallest(k, "score")

    if time_col and any(word in q for word in ["time", "trend", "latest", "earliest"]):
        work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
        timed = work[work[time_col].notna()].sort_values(time_col)
        if not timed.empty:
            if "latest" in q:
                return timed.tail(k)
            if "earliest" in q:
                return timed.head(k)
            # For trends, provide both ends of the series.
            half = max(1, k // 2)
            return pd.concat([timed.head(half), timed.tail(half)], axis=0)

    if any(word in q for word in ["highest", "max", "warmest"]):
        return work.nlargest(k, temp_col)
    if any(word in q for word in ["lowest", "min", "coldest"]):
        return work.nsmallest(k, temp_col)

    return work.sample(min(k, len(work)), random_state=42)


def format_context_for_prompt(context_rows: pd.DataFrame, max_rows: int = 25) -> str:
    cols = list(context_rows.columns)
    rows = context_rows.head(max_rows).to_dict(orient="records")
    return f"Columns: {cols}\nSample rows:\n{rows}"


def llm_with_openai(system_prompt: str, user_prompt: str) -> str:
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError(
            "OpenAI package not installed. Run: pip install openai"
        ) from None

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
    )
    return response.choices[0].message.content or "No response from OpenAI."


def llm_with_ollama(system_prompt: str, user_prompt: str) -> str:
    model = os.getenv("OLLAMA_MODEL", "tinydolphin")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
    }
    response = requests.post(
        "http://localhost:11434/api/chat",
        json=payload,
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()
    return data.get("message", {}).get("content", "No response from Ollama.")


def answer_with_rag(df: pd.DataFrame, question: str, llm_backend: str) -> str:
    context_rows = retrieve_context_rows(df, question, k=40)
    context_text = format_context_for_prompt(context_rows, max_rows=30)

    dataset_summary = {
        "row_count": len(df),
        "columns": list(df.columns),
    }

    system_prompt = (
        "You are Float-Chat, an ocean data assistant. "
        "Answer using only the provided context. "
        "If data is insufficient, clearly say what is missing."
    )

    template = """
You are a helpful oceanographic assistant. Use the following data to answer the user's question. 
CRITICAL: Do NOT show the raw data, 'Dataset summary', or 'Sample rows' in your response. 
Just give a natural, concise answer.

Context: {context}
Question: {question}
Answer:"""

    combined_context = f"Dataset summary: {dataset_summary}\n\nRetrieved context:\n{context_text}"
    user_prompt = template.format(context=combined_context, question=question)

    try:
        if llm_backend == "OpenAI":
            return llm_with_openai(system_prompt, user_prompt)
        return llm_with_ollama(system_prompt, user_prompt)
    except Exception as exc:
        return (
            "RAG pipeline could not get an LLM response.\n"
            f"Reason: {exc}\n\n"
            "Fix:\n"
            "- For OpenAI: set OPENAI_API_KEY and install `openai`\n"
            "- For local: run Ollama and `ollama pull llama3.1`"
        )


with st.sidebar:
    st.title("🌊 Team Sphinx")
    st.caption("HackToFuture 4.0 | ARGO Intelligence")
    st.markdown("---")
    st.subheader("📡 System Health")
    st.success("Edge Node: Active")
    st.info("Agent: TinyDolphin")
    st.caption("Offline-capable semantic retrieval layer")
    st.markdown("---")
    llm_backend = st.selectbox("LLM Interface", ["Ollama (local)", "OpenAI"])
    backend_value = "OpenAI" if llm_backend == "OpenAI" else "Ollama"
    if backend_value == "OpenAI":
        st.caption("Requires `OPENAI_API_KEY`.")
    else:
        st.caption("Uses local Ollama at `localhost:11434`.")
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.title("Float-Chat: Autonomous Ocean Intelligence")
st.write("Democratizing scientific data through Edge-AI and semantic retrieval.")

try:
    data = load_data(CSV_PATH)
    with st.expander("🔍 Live Ground-Truth Monitor", expanded=False):
        st.dataframe(data.head(15), use_container_width=True)
except Exception as exc:
    st.error(f"Data Error: {exc}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "RAG pipeline operational. Ask me about ocean temperature trends."
            ),
        }
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_question = st.chat_input("Ask about ocean temperature from ARGO data...")

if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating answer..."):
            response = answer_with_rag(data, user_question, backend_value)
            st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
