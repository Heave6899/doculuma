import streamlit as st
from gpt4all import GPT4All
import openai
import google.generativeai as genai

# --- Local LLM (GPT4All) Handler ---
def _generate_with_gpt4all(prompt: str, model_path: str) -> str:
    """Generates a response using a local GPT4All model."""
    try:
        with GPT4All(model_path, device="cpu", allow_download=False) as model:
            with model.chat_session():
                response = model.generate(prompt=prompt, max_tokens=512, temp=0.2)
        return response
    except Exception as e:
        st.error(f"Error with local LLM: {e}")
        return f"Error: {e}"

# --- Online LLM (OpenAI) Handler ---
def _generate_with_openai(prompt: str, api_key: str, model: str = "gpt-3.5-turbo") -> str:
    """Generates a response using the OpenAI API."""
    if not api_key:
        st.error("OpenAI API key is not set. Please add it on the Settings page.")
        return "Error: API key missing."
    try:
        openai.api_key = api_key
        response = openai.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error with OpenAI API: {e}")
        return f"Error: {e}"

def _generate_with_gemini(prompt: str, api_key: str, model: str = "gemini-pro") -> str:
    """Generates a response using the Google Gemini API."""
    if not api_key:
        st.error("Google AI API key is not set. Please add it on the Settings page.")
        return "Error: API key missing."
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error with Google Gemini API: {e}")
        return f"Error: {e}"

# --- Main LLM Manager Class (Updated) ---
class LLMManager:
    """A manager to handle interactions with different LLM providers."""
    def __init__(self):
        # Load settings from session state
        self.provider = st.session_state.get("llm_provider", "Local")
        self.local_model_path = st.session_state.get("local_model_path", "sqlcoder-7b-q5_k_m.gguf")
        self.openai_api_key = st.session_state.get("openai_api_key", "")
        self.google_api_key = st.session_state.get("google_api_key", "") # <-- Add Google API key

    def generate(self, prompt: str) -> str:
        """
        Generates a response using the currently configured LLM provider.
        """
        st.info(f"🧠 Using **{self.provider}** LLM for this task...")
        if self.provider == "Local":
            return _generate_with_gpt4all(prompt, self.local_model_path)
        elif self.provider == "OpenAI":
            return _generate_with_openai(prompt, self.openai_api_key)
        elif self.provider == "Google Gemini": # <-- Add Gemini provider
            return _generate_with_gemini(prompt, self.google_api_key, 'gemini-2.0-flash')
        else:
            return "Error: Unknown LLM provider selected."