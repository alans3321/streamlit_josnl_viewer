import streamlit as st
import pandas as pd
import io
import json
import os
from typing import Optional

st.set_page_config(page_title="JSONL Annotator", layout="wide")

st.title("Lost in Translation Annotations")
st.markdown(
    "Upload the  JSONL file provided. The model reasons in English for a non English question and has resulted in an incorrect answer. Please find out whether the incorrect answer is due to reasoning in English getting lost in translations i.e., mistranslations leading to incorrect answer. If it is 'lost in translation' mark  `Yes` else `No`.\n\n"
    "Once done download the output jsonl file and share it with Alan ."
)
    # "Upload a JSONL file (one JSON object per line). Each object should contain the fields shown in your example (e.g. `custom_id`, `input prompt`, etc.). The app shows one item at a time and lets an annotator mark `Yes` or `No`.\n\n"
    # "Annotations are saved to `annotated_output.jsonl` and `annotated_output.csv` in the working directory and also available for download."
# Helper functions

def load_jsonl_from_bytes(b: bytes) -> pd.DataFrame:
    text = b.decode("utf-8")
    lines = [line for line in text.splitlines() if line.strip()]
    records = [json.loads(line) for line in lines]
    return pd.DataFrame(records)


def save_jsonl(df: pd.DataFrame, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in df.to_dict(orient="records"):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# File uploader or load example
uploaded = st.sidebar.file_uploader("Upload JSONL file", type=["jsonl", "txt", "json"]) 
use_example = st.sidebar.checkbox("Use example data (demo)")

if "df" not in st.session_state:
    st.session_state.df = None
    st.session_state.idx = 0

if uploaded is not None:
    try:
        st.session_state.df = load_jsonl_from_bytes(uploaded.getvalue())
        st.success(f"Loaded {len(st.session_state.df)} records from uploaded file")
    except Exception as e:
        st.error(f"Failed to parse uploaded file: {e}")

elif use_example:
    # Minimal example (keeps content short). Replace with real file if needed.
    example = {
        "custom_id": "Qwen_QwQ-32B_trial-4_lang_hi-doc_id_65-classification",
        "input prompt": "Here is a chain-of-reasoning... <think>So, Jim watches TV for 2 hours... </think>",
        "input": "(full original JSON/ prompt here)"
    }
    st.session_state.df = pd.DataFrame([example])

# If no file yet, show instruction and stop
if st.session_state.df is None:
    st.info("Upload a JSONL file on the left (or enable 'Use example data') to begin annotation.")
    st.stop()

# Ensure annotation column exists
if "annotation" not in st.session_state.df.columns:
    st.session_state.df["annotation"] = ""
if "annotator_comment" not in st.session_state.df.columns:
    st.session_state.df["annotator_comment"] = ""

# Sidebar controls for navigation
n = len(st.session_state.df)
st.sidebar.markdown(f"**Records:** {n}")
idx = st.sidebar.number_input("Go to index (0-based)", min_value=0, max_value=max(0, n-1), value=st.session_state.idx, step=1)
st.session_state.idx = int(idx)

col_prev, col_next = st.sidebar.columns(2)
if col_prev.button("◀ Prev"):
    st.session_state.idx = max(0, st.session_state.idx - 1)

if col_next.button("Next ▶"):
    st.session_state.idx = min(n-1, st.session_state.idx + 1)

st.sidebar.markdown("---")
if st.sidebar.button("Save all annotations now"):
    save_jsonl(st.session_state.df, "annotated_output.jsonl")
    st.session_state.df.to_csv("annotated_output.csv", index=False)
    st.sidebar.success("Saved annotated_output.jsonl and annotated_output.csv")

# Main display for the current record
rec = st.session_state.df.iloc[st.session_state.idx]
st.subheader(f"Record {st.session_state.idx} / {n-1}")

# Show available fields in a readable layout
with st.expander("View full record (raw JSON)", expanded=False):
    st.json(rec.to_dict())

# Try to extract 'input prompt' or 'input' or fallback to raw
prompt_text = None
for key in ["input prompt", "input", "prompt", "text"]:
    if key in st.session_state.df.columns:
        prompt_text = rec.get(key)
        break

if prompt_text is None:
    # fallback: render the entire record as text
    prompt_text = json.dumps(rec.to_dict(), ensure_ascii=False, indent=2)

# Present the prompt and chain-of-thought (if present in tags)
st.markdown("**Prompt**")
st.write(prompt_text)

# If there's a chain-of-reasoning demarcated by <think>...</think>, show it separately
if isinstance(prompt_text, str) and "<think>" in prompt_text and "</think>" in prompt_text:
    start = prompt_text.find("<think>") + len("<think>")
    end = prompt_text.find("</think>")
    chain = prompt_text[start:end].strip()


# Annotation widgets
st.markdown("---")
col1, col2 = st.columns([2, 1])
with col1:
    annot = st.radio("Annotator label: Has the reasoning thread got 'Lost in Translation?", ("Yes", "No"), index=0 if rec.get("annotation") in ["Yes","yes","YES"] else 1 if rec.get("annotation") in ["No","no","NO"] else 2 if rec.get("annotation") else 0)
    comment = st.text_area("Annotator comment (optional)", value=rec.get("annotator_comment", ""), height=120)

with col2:
    if st.button("Save / Update this record"):
        st.session_state.df.at[st.session_state.idx, "annotation"] = annot
        st.session_state.df.at[st.session_state.idx, "annotator_comment"] = comment
        save_jsonl(st.session_state.df, "annotated_output.jsonl")
        st.session_state.df.to_csv("annotated_output.csv", index=False)
        st.success("Saved annotation for current record and exported files.")

    # if st.button("Mark as Skip and go Next"):
    #     st.session_state.idx = min(n-1, st.session_state.idx + 1)
    #     st.experimental_rerun()

# Quick actions
st.markdown("---")
col_a, col_b = st.columns(2)
with col_a:
    csv = st.session_state.df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV of annotations", data=csv, file_name="annotated_output.csv", mime="text/csv")
with col_b:
    jsonl_bytes = "\n".join([json.dumps(r, ensure_ascii=False) for r in st.session_state.df.to_dict(orient="records")]).encode("utf-8")
    st.download_button("Download JSONL of annotations", data=jsonl_bytes, file_name="annotated_output.jsonl", mime="application/json")

# Show a compact table overview
st.markdown("### Annotation progress")
show_df = st.session_state.df.copy()
if show_df.shape[0] > 200:
    st.write("(Showing first 200 rows)")
    show_df = show_df.head(200)
st.dataframe(show_df)

st.caption("Files saved to the app working directory: annotated_output.jsonl and annotated_output.csv")
