import streamlit as st
import pandas as pd
from typing import List, Optional

def get_enhanced_db_schema(conn, tables: Optional[List[str]] = None) -> str:
    """
    Returns a detailed, line-by-line schema of the database tables using PRAGMA,
    including column types and primary keys.
    """
    if tables is None:
        # If no specific tables are requested, get all user-created tables
        query = "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        tables = [row[0] for row in conn.execute(query).fetchall()]

    lines = []
    for table_name in tables:
        try:
            # Use PRAGMA for reliable schema info
            columns_info = conn.execute(f'PRAGMA table_info("{table_name}");').fetchall()
            # columns_info is a list of tuples: (cid, name, type, notnull, dflt_value, pk)
            
            col_details = []
            for col in columns_info:
                name, dtype, pk = col[1], col[2], col[5]
                detail = f"{name} ({dtype})"
                if pk == 1:
                    detail += " PRIMARY KEY"
                col_details.append(detail)
            
            lines.append(f"Table {table_name}: {', '.join(col_details)}")
        except Exception as e:
            st.warning(f"Could not get schema for table {table_name}: {e}")
            
    return "\n".join(lines)

def get_table_sample(conn, table: str, n: int = 3) -> str:
    """Gets a markdown-formatted sample of rows from a table."""
    try:
        # Use read_sql_query for safety and compatibility
        df = pd.read_sql_query(f'SELECT * FROM "{table}" LIMIT {n}', conn)
        # Return an empty string if no data, otherwise return the markdown table
        return f"\n-- Top 3 rows from {table}:\n{df.to_markdown(index=False)}\n" if not df.empty else ""
    except Exception as e:
        st.warning(f"Could not get samples for table {table}: {e}")
        return ""

def get_db_schema_for_prompt(conn, selected_tables: Optional[List[str]] = None, with_samples: bool = True) -> str:
    """
    Constructs the full schema and sample rows prompt for the LLM using the enhanced methods.
    """
    # 1. Get the list of tables to process
    tables_to_process = selected_tables
    if not tables_to_process:
        query = "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        tables_to_process = [row[0] for row in conn.execute(query).fetchall()]

    # 2. Get the enhanced schema for those tables
    schema = get_enhanced_db_schema(conn, tables_to_process)
    
    # 3. Get samples if requested, using the reliable table list
    if with_samples:
        samples = "".join(get_table_sample(conn, table) for table in tables_to_process)
        return f"{schema}{samples}"
        
    return schema