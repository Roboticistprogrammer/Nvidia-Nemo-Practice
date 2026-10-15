"""Shared helpers for all pages: backend calls, the API-key sidebar and the model picker.

The user's NVIDIA API key lives only in this browser session's state (server memory of the
Streamlit process); it is sent to the backend with each request and never written to disk.
"""
import os

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = (os.getenv("BACKEND_URL") or "http://localhost:8000").rstrip("/")
TIMEOUT_S = 600
KEY_HEADER = "X-NVIDIA-API-Key"


def _headers():
    key = st.session_state.get("nvidia_api_key")
    return {KEY_HEADER: key} if key else {}


def api(method, path, **kwargs):
    """Call the backend with the user's key attached."""
    kwargs.setdefault("timeout", TIMEOUT_S)
    return requests.request(method, f"{BACKEND_URL}{path}", headers=_headers(), **kwargs)


def error_detail(resp):
    try:
        return resp.json().get("detail", resp.text)
    except ValueError:
        return resp.text[:300] or f"HTTP {resp.status_code}"


def _server_key_available():
    if "server_key_available" not in st.session_state:
        try:
            resp = requests.get(f"{BACKEND_URL}/auth/config", timeout=10)
            st.session_state["server_key_available"] = resp.ok and resp.json().get("server_key_available", False)
        except requests.RequestException:
            st.error(f"Cannot reach the backend at {BACKEND_URL}.")
            st.stop()
    return st.session_state["server_key_available"]


def _verify(key):
    """Returns (workspace, error)."""
    try:
        resp = requests.post(f"{BACKEND_URL}/auth/verify", headers={KEY_HEADER: key} if key else {}, timeout=30)
    except requests.RequestException:
        return None, f"Cannot reach the backend at {BACKEND_URL}."
    if resp.ok:
        return resp.json()["workspace"], None
    return None, error_detail(resp)


def require_api_key():
    """Render the API-key box in the sidebar and stop the page until a working key is set."""
    with st.sidebar:
        st.markdown("### NVIDIA API key")
        if st.session_state.get("workspace"):
            who = "your key" if st.session_state.get("nvidia_api_key") else "the server's key"
            st.success(f"Connected with {who} · workspace `{st.session_state['workspace']}`")
            if st.session_state.get("nvidia_api_key") and st.button("Sign out"):
                for k in ("nvidia_api_key", "workspace"):
                    st.session_state.pop(k, None)
                st.rerun()
            return

        with st.form("api_key_form"):
            key = st.text_input("Your key", type="password", placeholder="nvapi-...")
            submitted = st.form_submit_button("Connect")
        st.caption("Get a free key at [build.nvidia.com](https://build.nvidia.com). It is only kept for "
                   "this browser session; your documents are stored in a private workspace tied to it.")
        if submitted:
            workspace, err = _verify(key.strip())
            if err:
                st.error(err)
            else:
                st.session_state["nvidia_api_key"] = key.strip()
                st.session_state["workspace"] = workspace
                st.rerun()

    if _server_key_available():
        workspace, err = _verify("")
        if workspace:
            st.session_state["workspace"] = workspace
            st.rerun()
    st.info("Enter your NVIDIA API key in the sidebar to get started.")
    st.stop()


def model_selector(prefix):
    resp = api("GET", f"/{prefix}/get-models/")
    models = resp.json()["models"] if resp.ok else []
    if not models:
        st.error("Error fetching models.")
    with st.sidebar:
        return st.selectbox("Choose an LLM", models)
