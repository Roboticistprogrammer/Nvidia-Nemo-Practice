# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time

import streamlit as st

from common import api, error_detail, model_selector, require_api_key

st.set_page_config(layout="wide")
require_api_key()

st.title("Chat with your Knowledge Graph!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

llm = model_selector("chat")
with st.sidebar:
    use_kg = st.checkbox("Use knowledge graph")

user_input = st.chat_input("Can you tell me how research helps users to solve problems?")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # Send the user input to the backend for processing
        with st.spinner("Thinking..."):
            response = api("POST", "/chat/chat/", json={"user_input": user_input, "use_kg": use_kg, "model_id": llm})
        if response.ok:
            assistant_response = response.json().get("assistant_response", "")
            if "knowledge graph is currently unavailable" in assistant_response:
                st.error("No knowledge graph yet. Upload and process your PDFs on the main page first.")
            else:
                for chunk in assistant_response.split(' '):
                    full_response += chunk + ' '
                    message_placeholder.markdown(full_response)
                    time.sleep(0.05)
        else:
            st.error(f"Error processing the chat request: {error_detail(response)}")

    if full_response:
        st.session_state.messages.append({"role": "assistant", "content": full_response})
