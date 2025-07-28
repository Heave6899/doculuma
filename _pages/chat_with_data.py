# _pages/chat_with_data.py
import streamlit as st
import pandas as pd
import json
import db_utils
import llm_utils  # MODIFIED: Use your provided llm_utils
import config
from llm_manager import LLMManager # MODIFIED: Use your provided LLMManager

# --- Tool Execution Functions (No changes here) ---
def execute_sql_tool(conn, sql_query: str):
    """Executes a SQL query and returns the result as a DataFrame."""
    try:
        df = pd.read_sql_query(sql_query, conn)
        return df, None
    except Exception as e:
        return None, f"Error executing SQL: {e}"

def execute_semantic_tool(query: str):
    """Executes a semantic search using ChromaDB."""
    try:
        client = db_utils.get_chroma_client()
        results = db_utils.semantic_search_chroma(client, query, config.CHROMA_COLLECTION_PREFIX)
        return results, None
    except Exception as e:
        return None, f"Error during semantic search: {e}"

# --- Main Page Rendering Function ---
def render(conn):
    st.header("💬 Chat with Data")
    st.caption("Ask questions about your uploaded data. The assistant can use SQL and Semantic Search to find answers.")

    # MODIFIED: Initialize the manager from session state. No API key input needed here.
    try:
        llm_manager = LLMManager()
        if not st.session_state.get("llm_provider"):
             st.warning("Please configure your LLM provider on the Settings page first.")
             return
    except Exception as e:
        st.error(f"Failed to initialize LLM Manager: {e}")
        return

    # NEW: Allow user to select which tables the LLM should know about for this query.
    # This leverages your get_db_schema_for_prompt function's `selected_tables` parameter.
    all_tables = [t[0] for t in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'").fetchall()]
    with st.expander("Context Selection"):
        selected_tables = st.multiselect(
            "Select tables to include in context for the LLM",
            options=all_tables,
            default=all_tables
        )
        include_samples = st.checkbox("Include 3 sample rows per table in context", value=True)


    # Initialize chat history
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    # Display chat messages
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "details" in message:
                with st.expander("Show Details"):
                    st.json(message["details"])

    # Accept user input
    if prompt := st.chat_input("Ask about your data..."):
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # 1. First LLM call: Choose the right tool
                # MODIFIED: Use your enhanced schema prompter from llm_utils.py
                db_schema = llm_utils.get_db_schema_for_prompt(conn, selected_tables, with_samples=include_samples)
                
                tool_selection_prompt = f"""You are an expert data analyst assistant. Your goal is to help users by answering questions about their data.
You have two tools at your disposal:
1. `sql_query`: Use this for questions that require precise data, counting, filtering, or aggregating from a database.
2. `semantic_search`: Use this for broad, conceptual, or "what is" questions where the answer is likely in the text descriptions of the data.

Here is the schema of the tables you can query with `sql_query`:
{db_schema}

User question: "{prompt}"

Based on the user's question and the provided schema, choose the best tool. Respond with ONLY a single, valid JSON object with two keys: "tool" (either "sql_query" or "semantic_search") and "query" (the SQLite query or the search term)."""

                # MODIFIED: Use your manager's generate() method. The prompt is now a single block.
                tool_choice_str = llm_manager.generate(tool_selection_prompt)
                
                try:
                    # Clean the response to ensure it's valid JSON
                    tool_choice_json = re.search(r'\{.*\}', tool_choice_str, re.DOTALL).group(0)
                    tool_choice = json.loads(tool_choice_json)
                    chosen_tool = tool_choice.get("tool")
                    query = tool_choice.get("query")
                    
                    st.write(f"Tool Chosen: `{chosen_tool}`")
                    
                    # 2. Execute the chosen tool
                    if chosen_tool == "sql_query":
                        tool_result, error = execute_sql_tool(conn, query)
                    elif chosen_tool == "semantic_search":
                        tool_result, error = execute_semantic_tool(query)
                    else:
                        tool_result, error = (None, "Invalid tool chosen by the LLM.")

                    if error:
                        st.error(error)
                        response_details = {"error": str(error), "llm_tool_choice": tool_choice}
                        st.session_state.chat_messages.append({"role": "assistant", "content": f"Sorry, I encountered an error: {error}", "details": response_details})
                        st.rerun()

                    # 3. Second LLM call: Summarize the results
                    summarization_prompt = f"""A user asked the following question: "{prompt}"
To answer this, the '{chosen_tool}' tool was used with the query '{query}'.
The result of this tool is:
---
{tool_result}
---
Based ONLY on the data provided above, provide a concise, natural language answer to the user's original question."""
                    
                    # MODIFIED: Use your manager's generate() method again
                    final_answer = llm_manager.generate(summarization_prompt)
                    st.markdown(final_answer)

                    # Store the full context for transparency
                    response_details = {
                        "tool_used": chosen_tool,
                        "query_generated": query,
                        "raw_result": tool_result.to_dict() if isinstance(tool_result, pd.DataFrame) else tool_result,
                        "llm_tool_choice": tool_choice
                    }
                    st.session_state.chat_messages.append({"role": "assistant", "content": final_answer, "details": response_details})

                except (json.JSONDecodeError, AttributeError, KeyError) as e:
                    error_msg = f"Error processing LLM tool selection response: {e}. Raw response was: {tool_choice_str}"
                    st.error(error_msg)
                    st.session_state.chat_messages.append({"role": "assistant", "content": f"Sorry, I had trouble understanding the tool choice: {e}", "details": {"error": error_msg}})

            st.rerun()