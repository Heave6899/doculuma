import streamlit as st
import pandas as pd
import sqlite3
import json
import numpy as np
import db_utils
import llm_manager

# Helper functions
def execute_sql_on_row(row, sql_query):
    with sqlite3.connect(":memory:") as conn:
        row_df = pd.DataFrame([row])
        row_df.to_sql("row_table", conn, index=False)
        result = pd.read_sql_query(sql_query, conn)
    return result.to_dict(orient='records')


def execute_semantic_on_row(row, phrase, col, threshold):
    model = db_utils.get_embedding_model()
    phrase_emb = model.encode(phrase)
    row_emb = model.encode(str(row[col]))
    similarity = np.dot(phrase_emb, row_emb) / (np.linalg.norm(phrase_emb) * np.linalg.norm(row_emb))
    return similarity >= threshold


def execute_python_on_row(row, code):
    local_vars = {'row': row}
    exec(code, {}, local_vars)
    return local_vars.get('result', None)


def execute_llm_on_row(row, prompt_template):
    prompt = prompt_template.format(**row)
    llm_mgr = llm_manager.LLMManager()
    response = llm_mgr.generate(prompt)
    return response


# Streamlit UI & Logic
def render_loop_step(step, idx):
    st.subheader(f"🔄 Loop Step [{idx+1}]")

    if idx == 0:
        st.warning("Loop step requires at least one previous step.")
        return

    input_step_idx = st.selectbox(
        "🎯 Select Input Step:",
        options=list(range(idx)),
        key=f"loop_input_{step['id']}"
    )

    if input_step_idx is None:
        st.warning("Please select a valid input step.")
        return

    input_df = st.session_state.pipeline_steps[input_step_idx].get("output")

    if input_df is None or not isinstance(input_df, pd.DataFrame):
        st.warning("Selected input step has no valid output.")
        return

    action_step_type = st.selectbox(
        "🛠️ Action per Row:",
        ["SQL", "Semantic", "Python", "LLM"],
        key=f"action_{step['id']}"
    )

    if action_step_type == "SQL":
        sql_query = st.text_area("SQL Query (use 'row_table' as table name):", key=f"sql_{step['id']}")

    elif action_step_type == "Semantic":
        phrase = st.text_input("Semantic Phrase:", key=f"sem_phrase_{step['id']}")
        col = st.selectbox("Column for Semantic Match:", input_df.columns, key=f"sem_col_{step['id']}")
        threshold = st.slider("Similarity Threshold", 0.5, 1.0, 0.75, key=f"sem_threshold_{step['id']}")

    elif action_step_type == "Python":
        code = st.text_area("Python code (use 'row' as dict, store output in 'result'):", key=f"python_{step['id']}")

    elif action_step_type == "LLM":
        prompt_template = st.text_area("LLM Prompt (use placeholders like {col}):", key=f"llm_prompt_{step['id']}")

    if st.button("Execute Loop Step", key=f"exec_loop_{step['id']}"):
        results = []
        for _, row in input_df.iterrows():
            row_dict = row.to_dict()
            if action_step_type == "SQL":
                result = execute_sql_on_row(row_dict, sql_query)
                results.extend(result)

            elif action_step_type == "Semantic":
                matched = execute_semantic_on_row(row_dict, phrase, col, threshold)
                if matched:
                    results.append(row_dict)

            elif action_step_type == "Python":
                result = execute_python_on_row(row_dict, code)
                results.append({**row_dict, 'python_result': result})

            elif action_step_type == "LLM":
                response = execute_llm_on_row(row_dict, prompt_template)
                results.append({**row_dict, 'llm_response': response})

        step["output"] = pd.DataFrame(results)
        st.success("Loop step completed successfully.")
        st.dataframe(step["output"])
