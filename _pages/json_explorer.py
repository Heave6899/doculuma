import streamlit as st
import pandas as pd
import json
import json_utils
import io

def render(conn):
    st.header("4️⃣ Enhanced JSON Explorer & Diagrams")

    uploaded_files = st.file_uploader(
        "Upload JSON or JSON Schema files", type="json", accept_multiple_files=True
    )

    if uploaded_files:
        file_options = {f.name: f for f in uploaded_files}
        selected_file_name = st.selectbox("Select a JSON file to explore:", list(file_options.keys()))
        selected_file = file_options[selected_file_name]

        st.markdown("---")
        st.subheader(f"📁 Exploring: `{selected_file_name}`")

        try:
            selected_file.seek(0)
            data = json.load(selected_file)

            view_option = st.radio(
                "Select View",
                ["🌳 JSON Tree", "📊 Flattened Data Table", "📐 Schema Explorer"],
                horizontal=True
            )

            if view_option == "🌳 JSON Tree":
                st.markdown("#### Interactive JSON Tree View")
                search_query = st.text_input("🔍 Search JSON keys:")
                if search_query:
                    matches = json_utils.find_keys_recursive(data, search_query)
                    st.write(f"Matches for '{search_query}':", matches)
                st.json(data, expanded=True)

                

            elif view_option == "📊 Flattened Data Table":
                st.markdown("#### Flattened JSON Table")
                df_flat = pd.json_normalize(data, sep=".")
                st.dataframe(df_flat, use_container_width=True)

                csv_buffer = io.StringIO()
                df_flat.to_csv(csv_buffer, index=False)
                st.download_button(
                    "📥 Download CSV",
                    csv_buffer.getvalue(),
                    f"{selected_file_name}_flattened.csv",
                    "text/csv",
                )

            elif view_option == "📐 Schema Explorer":
                if isinstance(data, dict) and any(k in data for k in ["properties", "definitions", "components"]):
                    st.markdown("#### Schema Definition Table")
                    df_schema = json_utils.flatten_json_schema(data)

                    filter_col = st.text_input("Filter schema properties (by name):")
                    if filter_col:
                        df_schema = df_schema[df_schema['field'].str.contains(filter_col, case=False)]

                    st.dataframe(df_schema, use_container_width=True)

                    csv_buffer = io.StringIO()
                    df_schema.to_csv(csv_buffer, index=False)
                    st.download_button(
                        "📥 Download Schema CSV",
                        csv_buffer.getvalue(),
                        f"{selected_file_name}_schema.csv",
                        "text/csv",
                    )

                    st.markdown("#### Advanced Class Diagram")
                    dot_string = json_utils.generate_advanced_class_diagram_dot(data)
                    st.graphviz_chart(dot_string)
                else:
                    st.warning("This file doesn't appear to contain a JSON schema.")

        except json.JSONDecodeError as e:
            st.error(f"❌ JSON decode error in `{selected_file_name}`: {e}")
        except Exception as e:
            st.error(f"❌ Unexpected error processing `{selected_file_name}`: {e}")
    else:
        st.info("👆 Upload one or more JSON files to begin exploring.")
