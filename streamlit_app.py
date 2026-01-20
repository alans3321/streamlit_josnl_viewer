import streamlit as st
import pandas as pd
import json
from typing import Optional

st.set_page_config(page_title="Cognitive Behavior Annotator", layout="wide")

st.title("Cognitive Behavior Annotations")
st.markdown(
    "Upload the JSONL file provided. "
    "Count the occurrences of cognitive behaviors like  sub_goal_setting, verification, backtracking, and backward chaining in the reasoning trace."
    " Subgoal setting: Evaluate whether this reasoning explicitly sets any subgoals (e.g., “First I will try to isolate x…”, “Next I aim to simplify …”, etc.) on the way towards the final answer. List the number of such instances."
    " Verification: Evaluate whether this chain-of-reasoning contains any explicit answer-verification steps. An answer-verification step is any instance where the model checks its intermediate numeric result and ask itself if the answer is correct or not and probably go on to re check it. List the number of such steps."
    " Backtracking: Evaluate whether this reasoning contains any backtracking behavior—i.e., places where the model decides that its previous approach won’t reach the correct answer and explicitly abandons that path, starting fresh on an alternative intermediate step. List the number of such instances."
    " Backward Chaining: Evaluate whether this reasoning uses backward-chaining—i.e., it starts from the final answer and works backward to earlier steps. Count how many distinct backward-chaining instances occur."
    " Enter the counts below and save the record. Finally, download the output JSONL file."
)

# ---------- helper functions ----------
def load_jsonl_from_bytes(b: bytes) -> pd.DataFrame:
    text = b.decode("utf-8")
    lines = [line for line in text.splitlines() if line.strip()]
    records = [json.loads(line) for line in lines]
    return pd.DataFrame(records)

def save_jsonl(df: pd.DataFrame, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        # Convert to dict and ensure integers are written as standard numbers, not numpy types
        records = df.to_dict(orient="records")
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

# ---------- sidebar: load file or example ----------
uploaded = st.sidebar.file_uploader("Upload JSONL file", type=["jsonl", "txt", "json"])
use_example = st.sidebar.checkbox("Use example data (demo)")

if "df" not in st.session_state:
    st.session_state.df = None
if "idx" not in st.session_state:
    st.session_state.idx = 0

# List of columns we need for annotation
ANNOTATION_COLS = ["sub_goal_setting", "verification", "backtracking", "backward_chaining"]

# Load uploaded file (only once)
if uploaded is not None and st.session_state.df is None:
    try:
        st.session_state.df = load_jsonl_from_bytes(uploaded.getvalue())
        st.success(f"Loaded {len(st.session_state.df)} records from uploaded file: {uploaded.name}")
    except Exception as e:
        st.error(f"Failed to parse uploaded file: {e}")

elif use_example and st.session_state.df is None:
    example = {
        "custom_id": "demo_id_001",
        "input prompt": "Solve for x: 2x + 4 = 10",
        "input": "...",
        # pre-fill example data to 0
        "sub_goal_setting": 0,
        "verification": 0,
        "backtracking": 0,
        "backward_chaining": 0
    }
    st.session_state.df = pd.DataFrame([example])
    st.success("Loaded example data")

if st.session_state.df is None:
    st.info("Upload a JSONL file on the left (or enable 'Use example data') to begin annotation.")
    st.stop()

# Ensure annotation cols exist in DataFrame, default to 0 if missing
for col in ANNOTATION_COLS:
    if col not in st.session_state.df.columns:
        st.session_state.df[col] = 0

if "annotator_comment" not in st.session_state.df.columns:
    st.session_state.df["annotator_comment"] = ""

# Remove old 'annotation' column if it exists to keep output clean (optional)
if "annotation" in st.session_state.df.columns:
    st.session_state.df.drop(columns=["annotation"], inplace=True)

# ---------- navigation ----------
n = len(st.session_state.df)
st.sidebar.markdown(f"**Records:** {n}")

# use a widget key so the number_input value is preserved across reruns
idx_widget = st.sidebar.number_input(
    "Go to index (0-based)",
    min_value=0,
    max_value=max(0, n-1),
    value=st.session_state.idx,
    step=1,
    key="idx_input"
)
# sync the session_state idx with the widget
st.session_state.idx = int(idx_widget)

st.sidebar.markdown("---")

# ---------- main display ----------
rec = st.session_state.df.iloc[st.session_state.idx]
st.subheader(f"Record {st.session_state.idx} / {n-1}")

with st.expander("View full record (raw JSON)", expanded=False):
    st.json(rec.to_dict())

# find prompt text
prompt_text = None
for key in ["input prompt", "input", "prompt", "text"]:
    if key in st.session_state.df.columns:
        prompt_text = rec.get(key)
        break
if prompt_text is None:
    prompt_text = json.dumps(rec.to_dict(), ensure_ascii=False, indent=2)

st.markdown("**Prompt / Input**")
st.info(prompt_text)

# ---------- annotation widgets ----------
st.markdown("---")
st.markdown("### Annotator Label: Share the number of each cognitive behavior")

col_form, col_save = st.columns([3, 1])

with col_form:
    # We use a 2x2 grid for the inputs to make it look cleaner
    r1_c1, r1_c2 = st.columns(2)
    r2_c1, r2_c2 = st.columns(2)

    # Helper to safely get int value
    def get_val(key):
        val = rec.get(key, 0)
        return int(val) if pd.notnull(val) and val != "" else 0

    with r1_c1:
        sub_goal_val = st.number_input(
            "Sub goal setting", 
            min_value=0, 
            step=1, 
            value=get_val("sub_goal_setting"),
            key=f"sub_{st.session_state.idx}"
        )
    with r1_c2:
        verification_val = st.number_input(
            "Verification", 
            min_value=0, 
            step=1, 
            value=get_val("verification"),
            key=f"ver_{st.session_state.idx}"
        )
    with r2_c1:
        backtracking_val = st.number_input(
            "Backtracking", 
            min_value=0, 
            step=1, 
            value=get_val("backtracking"),
            key=f"back_{st.session_state.idx}"
        )
    with r2_c2:
        backward_chaining_val = st.number_input(
            "Backward chaining", 
            min_value=0, 
            step=1, 
            value=get_val("backward_chaining"),
            key=f"chain_{st.session_state.idx}"
        )

    comment = st.text_area(
        "Annotator comment (optional)",
        value=rec.get("annotator_comment", ""),
        height=80,
        key=f"comment_{st.session_state.idx}"
    )

with col_save:
    st.write("##") # Spacer
    if st.button("Save / Update this record", type="primary"):
        # update DataFrame in session state
        st.session_state.df.at[st.session_state.idx, "sub_goal_setting"] = int(sub_goal_val)
        st.session_state.df.at[st.session_state.idx, "verification"] = int(verification_val)
        st.session_state.df.at[st.session_state.idx, "backtracking"] = int(backtracking_val)
        st.session_state.df.at[st.session_state.idx, "backward_chaining"] = int(backward_chaining_val)
        st.session_state.df.at[st.session_state.idx, "annotator_comment"] = comment
        
        # persist to disk
        save_jsonl(st.session_state.df, "annotated_output.jsonl")
        st.session_state.df.to_csv("annotated_output.csv", index=False)
        
        # reassign copy so Streamlit notices the mutation
        st.session_state.df = st.session_state.df.copy()
        
        st.success("✅ Saved!")

# ---------- downloads & table ----------
st.markdown("---")
col_a, col_b = st.columns(2)
with col_a:
    csv = st.session_state.df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV of annotations", data=csv, file_name="annotated_output.csv", mime="text/csv")
with col_b:
    # Ensure integers are serialized correctly in JSONL
    jsonl_bytes = "\n".join([json.dumps(r, ensure_ascii=False) for r in st.session_state.df.to_dict(orient="records")]).encode("utf-8")
    st.download_button("Download JSONL of annotations", data=jsonl_bytes, file_name="annotated_output.jsonl", mime="application/json")

st.markdown("### Annotation progress")
show_df = st.session_state.df.copy()
if show_df.shape[0] > 200:
    st.write("(Showing first 200 rows)")
    show_df = show_df.head(200)
st.dataframe(show_df)

st.caption("Files saved to the app working directory: annotated_output.jsonl and annotated_output.csv")
