# 🤖 AI CV Agent

> A conversational AI that brings a static resume to life — ask it anything about my experience, skills, and projects.

**Live Demo:** https://huggingface.co/spaces/xanderKariuki/ai-cv-agent

Note: This application is hosted on the Hugging Face Spaces CPU Basic (Free) tier. If the app appears unavailable or displays a scheduling/runtime error, it is usually due to temporary resource allocation on the shared free infrastructure rather than an issue with the application itself. Please try again in a few minutes or refresh the page.
---

## What It Does

Instead of handing someone a PDF and hoping they read it, this app lets anyone have a real conversation with an AI that knows my professional background in depth. It answers questions about work history, technical skills, and projects — and if someone wants to follow up, it captures their contact details automatically.

---

## How It Works

The app is built around three layers:

**Data**
At startup, the agent loads two source files — a LinkedIn profile export (`data/linkedin.pdf`) and a written professional summary (`data/summary.txt`). These are parsed and injected into the system prompt once, so every conversation is grounded in accurate, up-to-date information.

**Agent**
The core is an OpenAI `gpt-4o-mini` chat agent with two tools available to it:

- `record_user_details` — called when a visitor provides their email or asks to be contacted. Captures their name, email, and any relevant context, then fires a Pushover push notification in real time.
- `record_unknown_question` — called when the agent can't confidently answer a question. Logs it via Pushover so gaps in the source data can be identified and addressed.

The agent runs a tool-calling loop before streaming the final response, meaning tool calls are resolved silently in the background and the user only ever sees the natural reply — streamed token by token.

**UI**
Built with Gradio 5. The interface includes:
- A live streaming chatbot that renders responses incrementally
- Three quick-action buttons (Experience, Skills, Projects) that auto-submit without needing the textbox
- An About Me section pulled directly from the summary file

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | OpenAI `gpt-4o-mini` |
| Agent framework | OpenAI Python SDK (tool-calling) |
| UI | Gradio 5 |
| PDF parsing | pypdf |
| Notifications | Pushover |
| Deployment | HuggingFace Spaces |

---

## Project Structure

```
.
├── app.py                  # Main application
├── requirements.txt        # Python dependencies
└── data/
    ├── linkedin.pdf        # LinkedIn profile export
    └── summary.txt         # Written professional summary
```

---

## Running Locally

**1. Clone the repo**
```bash
git clone <repo-url>
cd <repo-name>
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Set up environment variables**

Create a `.env` file in the root directory:
```
OPENAI_API_KEY=your-openai-key
PUSHOVER_TOKEN=your-pushover-app-token
PUSHOVER_USER=your-pushover-user-key
```

Pushover credentials are optional — the app will skip notifications and log a warning if they are absent.

**4. Add your data files**
```
data/linkedin.pdf   ← export from LinkedIn (Settings → Data Privacy → Get a copy of your data)
data/summary.txt    ← a written bio or professional summary
```

**5. Run**
```bash
python app.py
```

Visit `http://localhost:7860` in your browser.

---

## Deploying to HuggingFace Spaces

1. Create a new Space with the **Gradio** SDK
2. Push this repository to the Space
3. Add the following under **Settings → Variables and Secrets**:
   - `OPENAI_API_KEY`
   - `PUSHOVER_TOKEN`
   - `PUSHOVER_USER`
4. Ensure `server_name="0.0.0.0"` is set in `demo.launch()` — required for Spaces to route traffic correctly

---

## Notes

- The agent never fabricates information. If a question falls outside what the source data covers, it logs the question and lets the visitor know honestly.
- No conversation history is stored server-side — each session is stateless.
- The data files are read once at startup, not on every message, keeping response latency low.
