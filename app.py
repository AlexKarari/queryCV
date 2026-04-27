"""
AI CV Agent — app.py
"""

import json
import os
import requests
import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
load_dotenv(override=True)

MODEL = "gpt-4o-mini"
MAX_TOOL_LOOPS = 5
SUMMARY_PREVIEW_LENGTH = 500
DATA_DIR = "data"
LINKEDIN_PATH = os.path.join(DATA_DIR, "linkedin.pdf")
SUMMARY_PATH = os.path.join(DATA_DIR, "summary.txt")

AGENT_NAME = "Alexander K. Kariuki"

# HuggingFace Spaces sets this env var automatically — used to toggle launch config
IS_HUGGINGFACE = os.getenv("SPACE_ID") is not None


# ---------------------------------------------------------------------------
# Notifications
# ---------------------------------------------------------------------------
def push(text: str) -> None:
    """Send a Pushover notification. Fails silently with a console warning."""
    token = os.getenv("PUSHOVER_TOKEN")
    user = os.getenv("PUSHOVER_USER")

    if not token or not user:
        print("[WARN] Pushover credentials missing — skipping notification.")
        return

    try:
        resp = requests.post(
            "https://api.pushover.net/1/messages.json",
            data={"token": token, "user": user, "message": text},
            timeout=5,
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"[WARN] Pushover notification failed: {e}")


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------
def record_user_details(email: str, name: str = "Name not provided", notes: str = "not provided") -> dict:
    push(f"New lead — {name} | {email} | Notes: {notes}")
    return {"recorded": "ok"}


def record_unknown_question(question: str) -> dict:
    push(f"Unknown question: {question}")
    return {"recorded": "ok"}


# ---------------------------------------------------------------------------
# Tool schemas (OpenAI format) — co-located with their Python functions above
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "record_user_details",
            "description": (
                "Call this tool when a user provides their email or asks to be "
                "contacted for follow-up."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "email": {
                        "type": "string",
                        "format": "email",
                        "description": "The user's valid email address.",
                    },
                    "name": {
                        "type": "string",
                        "description": "The user's name if explicitly provided.",
                    },
                    "notes": {
                        "type": "string",
                        "description": "Relevant context or summary of the conversation.",
                    },
                },
                "required": ["email"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "record_unknown_question",
            "description": (
                "Call this tool when a user asks a question you cannot confidently "
                "answer or lack sufficient information to respond accurately."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The exact user question that could not be answered.",
                    }
                },
                "required": ["question"],
                "additionalProperties": False,
            },
        },
    },
]

TOOL_REGISTRY = {
    "record_user_details": record_user_details,
    "record_unknown_question": record_unknown_question,
}


# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------
class DataLoader:
    """Loads and validates CV source files at startup."""

    def __init__(self, linkedin_path: str, summary_path: str):
        self.linkedin = self._load_pdf(linkedin_path)
        self.summary = self._load_text(summary_path)

    @staticmethod
    def _load_pdf(path: str) -> str:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"LinkedIn PDF not found at '{path}'. "
                "Ensure 'data/linkedin.pdf' exists before launching."
            )
        reader = PdfReader(path)
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted
        if not text.strip():
            raise ValueError(f"LinkedIn PDF at '{path}' appears to be empty or unreadable.")
        return text

    @staticmethod
    def _load_text(path: str) -> str:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Summary file not found at '{path}'. "
                "Ensure 'data/summary.txt' exists before launching."
            )
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        if not content.strip():
            raise ValueError(f"Summary file at '{path}' is empty.")
        return content


# ---------------------------------------------------------------------------
# Chat agent
# ---------------------------------------------------------------------------
class ChatAgent:
    """Handles OpenAI chat completions with tool-calling support."""

    def __init__(self, name: str, linkedin: str, summary: str):
        self.client = OpenAI()
        # Build the system prompt once — data never changes at runtime
        self._system_prompt = self._build_system_prompt(name, linkedin, summary)

    @staticmethod
    def _build_system_prompt(name: str, linkedin: str, summary: str) -> str:
        return (
            f"You are acting as {name}. You answer questions about {name}'s CV, "
            "experience, and skills.\n"
            "Be professional, engaging, and concise.\n"
            "If you don't know an answer, call the `record_unknown_question` tool.\n"
            "If the user shows interest in connecting, encourage them to share their "
            "email and capture it using the `record_user_details` tool.\n\n"
            f"## Summary\n{summary}\n\n"
            f"## LinkedIn Profile\n{linkedin}\n\n"
            f"Stay in character as {name} at all times."
        )

    def _handle_tool_calls(self, tool_calls) -> list[dict]:
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            try:
                args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                result = {"error": "Could not parse tool arguments."}
            else:
                func = TOOL_REGISTRY.get(tool_name)
                if func is None:
                    result = {"error": f"Unknown tool: {tool_name}"}
                else:
                    try:
                        result = func(**args)
                    except Exception as e:
                        result = {"error": str(e)}

            results.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result),
            })
        return results

    def chat(self, user_message: str, history: list[dict]) -> str:
        """
        Args:
            user_message: The latest user input.
            history: List of {"role": ..., "content": ...} dicts (OpenAI format).
        Returns:
            Assistant reply string.
        """
        messages = (
            [{"role": "system", "content": self._system_prompt}]
            + history
            + [{"role": "user", "content": user_message}]
        )

        for _ in range(MAX_TOOL_LOOPS):
            response = self.client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOLS,
            )
            choice = response.choices[0]

            if choice.finish_reason == "tool_calls":
                tool_results = self._handle_tool_calls(choice.message.tool_calls)
                messages.append(choice.message)
                messages.extend(tool_results)
            else:
                return choice.message.content or "I'm not sure how to respond to that."

        print(f"[WARN] Tool loop hit MAX_TOOL_LOOPS ({MAX_TOOL_LOOPS}) without a final response.")
        return "I ran into an issue completing that request. Please try again."


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
def build_ui(agent: ChatAgent, summary_preview: str) -> gr.Blocks:

    def respond(user_message: str, history: list[dict]) -> tuple[str, list[dict]]:
        """
        Gradio 5.x chat handler.
        Returns a NEW history list — never mutates in place (Gradio 5 state requirement).
        """
        if not user_message.strip():
            return "", history
        reply = agent.chat(user_message, history)
        updated_history = history + [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": reply},
        ]
        return "", updated_history

    with gr.Blocks(theme=gr.themes.Soft(), title="AI CV Agent") as demo:

        # Header
        gr.Markdown(
            "# 🤖 AI CV Agent\n"
            "### Turn a static resume into a conversation\n\n"
            "Ask anything about my experience, skills, or projects.\n\n"
            "---"
        )

        # Quick-action buttons — instantiated inside their Row so layout is correct
        with gr.Row():
            btn_experience = gr.Button("💼 Experience")
            btn_skills = gr.Button("🧠 Skills")
            btn_projects = gr.Button("🚀 Projects")

        # Chat area — type="messages" is the Gradio 5 dict format
        chatbot = gr.Chatbot(height=450, type="messages")

        # Input row
        with gr.Row():
            msg = gr.Textbox(
                placeholder="Ask me anything...",
                show_label=False,
                scale=8,
            )
            send = gr.Button("Send", variant="primary", scale=1)

        # Text input handlers
        send.click(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])
        msg.submit(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])

        # Quick-action buttons — lambdas inject the fixed prompt, auto-submit immediately
        btn_experience.click(
            lambda h: respond("What experience do you have?", h),
            inputs=[chatbot],
            outputs=[msg, chatbot],
        )
        btn_skills.click(
            lambda h: respond("What are your key skills?", h),
            inputs=[chatbot],
            outputs=[msg, chatbot],
        )
        btn_projects.click(
            lambda h: respond("What projects have you worked on?", h),
            inputs=[chatbot],
            outputs=[msg, chatbot],
        )

        # About Me — string concatenation avoids f-string curly brace risk
        gr.Markdown(
            "---\n## 👤 About Me\n\n"
            + summary_preview
            + "\n\n---"
        )

        # Footer
        gr.Markdown("⚡ Built with LLMs, tool-calling, and Gradio")

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Add it as a HuggingFace Space secret "
            "or to your local .env file."
        )

    data = DataLoader(LINKEDIN_PATH, SUMMARY_PATH)
    agent = ChatAgent(
        name=AGENT_NAME,
        linkedin=data.linkedin,
        summary=data.summary,
    )

    # Safe word-boundary truncation for About Me preview
    if len(data.summary) > SUMMARY_PREVIEW_LENGTH:
        raw = data.summary[:SUMMARY_PREVIEW_LENGTH]
        summary_preview = raw.rsplit(" ", 1)[0] + "..."
    else:
        summary_preview = data.summary

    demo = build_ui(agent, summary_preview)

    # server_name="0.0.0.0" is required for HuggingFace Spaces.
    # Locally, access via http://localhost:7860 (not http://0.0.0.0:7860)
    demo.launch(server_name="0.0.0.0" if IS_HUGGINGFACE else "127.0.0.1")


if __name__ == "__main__":
    main()