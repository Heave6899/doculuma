import streamlit as st
import pandas as pd
import db_utils
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def clean_and_convert_column(series: pd.Series) -> pd.Series:
    """
    Cleans a text series containing financial data and converts it to numeric.
    """
    if series.dtype != 'object':
        return series
        
    # Clean string by removing $, ,, and handling () for negatives
    cleaned = series.str.replace('$', '', regex=False)
    cleaned = cleaned.str.replace(',', '', regex=False)
    cleaned = cleaned.str.replace(r'\((.*)\)', r'-\1', regex=True)
    
    # Convert to numeric, setting errors to null
    return pd.to_numeric(cleaned, errors='coerce')


def render(conn):
    """Renders the Data Visualization page."""
    st.header("📊 Data Visualization")

    try:
        all_tables = db_utils.list_tables(conn, 0)
        snapshot_tables = pd.read_sql_query("SELECT snapshot_table FROM versions", conn)["snapshot_table"].tolist()
        base_tables = [t for t in all_tables if t not in snapshot_tables and t != "versions"]
    except Exception:
        base_tables = []
    
    if not base_tables:
        st.warning("No base tables available for visualization. Please upload a file first.")
        return

    selected_table = st.selectbox("Select a table to visualize", options=[""] + base_tables)

    if not selected_table:
        st.info("Select a table to get started.")
        return

    df = pd.read_sql_query(f"SELECT * FROM `{selected_table}`", conn)
    st.dataframe(df.head(), use_container_width=True)

    st.markdown("---")
    st.subheader("Chart Configuration")

    chart_type = st.selectbox(
        "Select Chart Type",
        options=["Bar", "Line", "Scatter", "Word Cloud"]
    )
    
    # --- INTELLIGENT UPGRADE SECTION ---
    # Create a temporary, cleaned copy of the DataFrame for plotting
    df_for_plotting = df.copy()

    # Identify columns that are already numeric
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    
    # Identify text columns that LOOK like numbers (contain $, ,, or () )
    text_cols = df.select_dtypes(include="object").columns.tolist()
    cleanable_cols = [
        col for col in text_cols 
        if df[col].str.contains(r'[\$\(\),]', na=False).any()
    ]

    # Clean the "cleanable" columns in our temporary DataFrame
    for col in cleanable_cols:
        df_for_plotting[col] = clean_and_convert_column(df_for_plotting[col])
    
    # The list of plottable columns is now the original numerics + the cleaned ones
    plottable_numeric_cols = numeric_cols + cleanable_cols
    # --- END OF INTELLIGENT UPGRADE SECTION ---

    if chart_type in ["Bar", "Line", "Scatter"]:
        if not plottable_numeric_cols:
            st.error("This table contains no columns that are or look like numbers.")
            return
            
        c1, c2 = st.columns(2)
        with c1:
            x_axis = st.selectbox("Select X-Axis", options=df.columns.tolist(), key="xaxis_std")
        with c2:
            # The Y-axis dropdown now intelligently includes the cleaned columns
            y_axis = st.selectbox("Select Y-Axis (numeric or financial text)", options=plottable_numeric_cols, key="yaxis_std")
        
        if x_axis and y_axis:
            st.markdown("---")
            st.subheader(f"{chart_type} Chart for '{y_axis}'")
            try:
                # Use the cleaned DataFrame for plotting
                if chart_type == "Bar":
                    st.bar_chart(df_for_plotting, x=x_axis, y=y_axis)
                elif chart_type == "Line":
                    st.line_chart(df_for_plotting, x=x_axis, y=y_axis)
                elif chart_type == "Scatter":
                    st.scatter_chart(df_for_plotting, x=x_axis, y=y_axis)
            except Exception as e:
                st.error(f"Failed to create chart. Error: {e}")

    elif chart_type == "Word Cloud":
        if not text_cols:
            st.error("This table contains no text columns to generate a word cloud.")
            return

        text_column = st.selectbox("Select a text column for the Word Cloud", options=text_cols)

        if text_column:
            st.markdown("---")
            st.subheader(f"Word Cloud for '{text_column}'")
            text_data = " ".join(df[text_column].dropna().astype(str))
            
            with st.spinner("Generating Word Cloud..."):
                wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(text_data)
                fig, ax = plt.subplots()
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis("off")
                st.pyplot(fig)