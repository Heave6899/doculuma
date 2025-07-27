import streamlit as st
import config

def render(conn):
    """Renders the Settings page."""
    st.header("⚙️ Settings")
    

    # Initialize session state if it doesn't exist.
    st.session_state.setdefault("llm_provider", "Local")
    st.session_state.setdefault("local_model_path", config.LLM_MODEL_PATH)
    st.session_state.setdefault("openai_api_key", "")
    st.session_state.setdefault("google_api_key", "")

    with st.form(key="settings_form"):
        st.subheader("LLM Configuration")
        st.markdown("Select your provider and enter API keys if needed. Click 'Save Settings' at the bottom to apply.")
        
        provider_options = ["Local", "OpenAI", "Google Gemini"]
        current_provider_index = provider_options.index(st.session_state.llm_provider)

        provider = st.radio(
            "Select your LLM Provider",
            options=provider_options,
            index=current_provider_index,
            key="provider_radio",
            horizontal=True
        )

        if provider == "Local":
            st.text_input("Local Model Path", value=st.session_state.local_model_path, key="local_path_input")
        elif provider == "OpenAI":
            st.text_input("OpenAI API Key", value=st.session_state.openai_api_key, key="openai_key_input", type="password")
        elif provider == "Google Gemini":
            st.text_input("Google AI API Key", value=st.session_state.google_api_key, key="google_key_input", type="password")

        st.markdown("---")
        st.subheader("🔌 Integrations")

        st.session_state.setdefault("integrations", {})
        st.session_state.integrations.setdefault("unqork", False)

        st.session_state.integrations["unqork"] = st.checkbox(
            "Enable Unqork Integration", value=st.session_state.integrations["unqork"]
        )

        st.text_input("Unqork API Token", value=st.session_state.get("unqork_api_token", ""), key="unqork_api_token_input", type="password")
        st.session_state.integrations.setdefault("folder_import", False)

        st.session_state.integrations["folder_import"] = st.checkbox(
            "Enable Folder Import Integration", value=st.session_state.integrations["folder_import"]
        )
        submitted = st.form_submit_button("Save Settings")

    if submitted:
        st.session_state.llm_provider = st.session_state.provider_radio
        if provider == "Local":
            st.session_state.local_model_path = st.session_state.local_path_input
        elif provider == "OpenAI":
            st.session_state.openai_api_key = st.session_state.openai_key_input
        elif provider == "Google Gemini":
            st.session_state.google_api_key = st.session_state.google_key_input
        st.session_state["unqork_api_token"] = st.session_state["unqork_api_token_input"]
        st.toast("✅ Settings saved successfully!")

    st.info(f"**Current active provider:** `{st.session_state.llm_provider}`")