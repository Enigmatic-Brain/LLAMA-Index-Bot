import json
import re
import traceback

import requests
import streamlit as st

try:
    from pygments.lexers import guess_lexer
    from pygments.util import ClassNotFound
    PYGMENTS_AVAILABLE = True
except ImportError:
    PYGMENTS_AVAILABLE = False

st.set_page_config(
    page_title="LLM Chat",
    page_icon="💬",
    layout="wide",
)

# ── Code auto-detection ───────────────────────────────────────────────────────

# Patterns that strongly suggest code
_CODE_PATTERNS = re.compile(
    r"""
    (^\s{4,}|\t)                          # 4-space or tab indentation
    | (?:^|\s)(def|class|import|from|return|if|else|elif|for|while|try|except
               |function|const|let|var|=>|async|await
               |public|private|protected|static|void|int|str|bool
               |\#include|\#define|namespace|using)(?:\s|$|\()
    | [;{}]$                              # trailing ; { }
    | ->|::|===|!==|&&|\|\|              # common operators
    """,
    re.VERBOSE | re.MULTILINE,
)


def maybe_wrap_code(text: str) -> str:
    """
    If `text` looks like raw source code (no existing fences), wrap it in a
    fenced code block with an inferred language tag.
    Returns the original text unchanged if it doesn't look like code.
    """
    stripped = text.strip()

    # Already wrapped — leave it alone
    if stripped.startswith("```"):
        return text

    lines = stripped.splitlines()

    # Single-line inputs are almost never raw code pastes
    if len(lines) < 2:
        return text

    # Count lines that match code patterns
    matches = sum(1 for line in lines if _CODE_PATTERNS.search(line))
    ratio = matches / len(lines)

    # Need at least 30 % of lines to look like code before committing
    if ratio < 0.30:
        return text

    # Try to detect the language with pygments
    lang = ""
    if PYGMENTS_AVAILABLE:
        try:
            lexer = guess_lexer(stripped)
            # Only trust the guess if pygments is reasonably confident
            # (analysisresult score is not exposed directly; use alias as a
            # proxy — generic lexers like TextLexer have no useful aliases)
            aliases = lexer.aliases
            if aliases and aliases[0] not in ("text", "plain"):
                lang = aliases[0]
        except ClassNotFound:
            pass

    return f"```{lang}\n{stripped}\n```"


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Settings")
    api_key = st.text_input(
        "API Key",
        type="password",
        placeholder="Enter your API key",
        help="Your API key. It is never stored.",
    )
    endpoint = st.text_input(
        "Endpoint URL",
        placeholder="https://your-host/v1/chat/completions",
        help="Full URL of the chat completions endpoint, e.g. https://your-host/v1/chat/completions",
    )
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=0.7,
        step=0.05,
        help="Higher = more creative, lower = more deterministic.",
    )
    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── Page header ───────────────────────────────────────────────────────────────
st.title("LLM Chat Interface")

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Render existing messages ──────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat input ────────────────────────────────────────────────────────────────
prompt = st.chat_input("Ask anything… (paste code directly — it will be formatted automatically)")

if prompt:
    if not api_key:
        st.warning("Please enter your API key in the sidebar first.")
        st.stop()
    if not endpoint:
        st.warning("Please enter the endpoint URL in the sidebar first.")
        st.stop()

    # Auto-detect and wrap raw code pastes
    display_prompt = maybe_wrap_code(prompt)

    # Store the display version (with fences if code was detected) so the LLM
    # also sees it as a code block, which improves its responses
    st.session_state.messages.append({"role": "user", "content": display_prompt})
    with st.chat_message("user"):
        st.markdown(display_prompt)

    # Call the API and stream the response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        try:
            response = requests.post(
                endpoint,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "messages": st.session_state.messages,
                    "temperature": temperature,
                    "stream": True,
                },
                stream=True,
            )
            response.raise_for_status()

            for raw_line in response.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                delta = chunk["choices"][0].get("delta", {}).get("content") or ""
                full_response += delta
                response_placeholder.markdown(full_response + "▌")

            response_placeholder.markdown(full_response)

        except Exception as e:
            full_response = (
                f"**Error:** `{type(e).__name__}: {e}`\n\n"
                f"```\n{traceback.format_exc()}\n```"
            )
            response_placeholder.markdown(full_response)

    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )
