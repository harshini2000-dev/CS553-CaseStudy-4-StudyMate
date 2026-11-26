import os
import re
import gradio as gr
from huggingface_hub import InferenceClient

# Remote model only app.py
# removed local model for case study 4
# ============================================================
# CONFIGURATION
# ============================================================

REMOTE_MODEL = "Qwen/Qwen3-32B"

SUBJECTS = [
    "Machine Learning",
    "Data Science",
    "Deep Learning",
    "Statistics",
    "MLOps",
    "Computer Science",
    "General",
]


def clean_token(raw):
    """Strip whitespace and stray quote characters that sneak in from the shell."""
    if not raw:
        return None
    return raw.strip().strip('"').strip("'").strip()


HF_TOKEN = clean_token(os.environ.get("HF_TOKEN"))
print("HF_TOKEN found." if HF_TOKEN else "HF_TOKEN not set - chat will not work.")


# ============================================================
# PROMPT BUILDING
# ============================================================

def build_system_prompt(subject):
    return f"""You are StudyMate AI, a helpful academic study assistant.

The student's current subject is: {subject}

Your responsibilities:
- Explain concepts clearly and accurately.
- Use simple language when appropriate.
- Give examples when they help understanding.
- Help students learn rather than simply providing unexplained answers.
- Organize longer answers using headings or bullet points.
- Avoid unnecessary repetition.
- Do not mention that you are an AI unless it is relevant.
"""


def content_to_text(content):
    """Chatbot history content can be a string or a list of content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return ""


def build_messages(message, history, subject):
    messages = [{"role": "system", "content": build_system_prompt(subject)}]

    for msg in history:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        text = content_to_text(msg.get("content"))
        if role in ("user", "assistant") and text:
            messages.append({"role": role, "content": text})

    messages.append({"role": "user", "content": message})
    return messages


def strip_thinking(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# ============================================================
# MODEL CALL
# ============================================================

def remote_stream(messages, max_tokens, temperature, top_p):
    client = InferenceClient(model=REMOTE_MODEL, provider="auto", token=HF_TOKEN)
    response = ""
    for chunk in client.chat_completion(
        messages=messages,
        max_tokens=int(max_tokens),
        temperature=temperature,
        top_p=top_p,
        stream=True,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    ):
        choices = chunk.choices
        if len(choices) and choices[0].delta.content:
            response += choices[0].delta.content
            yield strip_thinking(response)


# ============================================================
# CHAT HANDLER
# ============================================================

def chat_and_update(message, history, subject, response_length, temperature, top_p):
    if not message or not message.strip():
        yield history, ""
        return

    messages = build_messages(message, history or [], subject)
    history = (history or []) + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": "⏳ Thinking..."},
    ]
    yield history, ""

    def set_reply(text):
        history[-1] = {"role": "assistant", "content": text}
        return history, ""

    if not HF_TOKEN or not HF_TOKEN.startswith("hf_"):
        yield set_reply(
            "⚠️ The server's `HF_TOKEN` is missing or invalid "
            "(it must start with `hf_`)."
        )
        return

    try:
        got_text = False
        for partial in remote_stream(messages, response_length, temperature, top_p):
            if partial:
                got_text = True
                yield set_reply(partial)

        if not got_text:
            yield set_reply(
                "Sorry, I couldn't generate a response. "
                "Please try asking the question again."
            )
    except Exception as e:
        yield set_reply(f"❌ Remote model failed:\n\n`{type(e).__name__}: {e}`")


def clear_chat():
    return [], ""


# ============================================================
# GRADIO USER INTERFACE
# ============================================================

with gr.Blocks(title="StudyMate AI") as demo:

    gr.Markdown(
        """
        # 🎓 StudyMate AI
        ### Your AI-Powered Study Assistant
        Ask questions, understand difficult concepts, and learn with a
        remotely hosted LLM.
        """
    )

    with gr.Row():

        # ---------------- Left: chat ----------------
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="💬 Conversation", height=500)

            message = gr.Textbox(
                label="Ask StudyMate",
                placeholder="Example: Explain overfitting in machine learning...",
                lines=3,
            )

            with gr.Row():
                send_button = gr.Button("🚀 Ask StudyMate", variant="primary")
                clear_button = gr.Button("🗑️ Clear Chat")

        # ---------------- Right: settings ----------------
        with gr.Column(scale=1):
            gr.Markdown("## ⚙️ Study Settings")

            subject = gr.Dropdown(
                choices=SUBJECTS,
                value="Machine Learning",
                label="📚 Subject",
            )

            response_length = gr.Slider(
                minimum=256, maximum=2048, value=1024, step=256,
                label="📝 Response Length (max new tokens)",
            )

            with gr.Accordion("🔧 Advanced", open=False):
                temperature = gr.Slider(
                    minimum=0.1, maximum=2.0, value=0.7, step=0.1,
                    label="Temperature",
                )
                top_p = gr.Slider(
                    minimum=0.1, maximum=1.0, value=0.95, step=0.05,
                    label="Top-p (nucleus sampling)",
                )

            gr.Markdown(
                f"""
                ---
                **Model:** `{REMOTE_MODEL}` via Hugging Face Inference Providers

                💡 **Tip:** Select a subject so StudyMate can tailor its
                explanations to your topic.
                """
            )

    # ---------------- Events ----------------
    chat_inputs = [message, chatbot, subject, response_length, temperature, top_p]

    send_button.click(chat_and_update, inputs=chat_inputs,
                      outputs=[chatbot, message])
    message.submit(chat_and_update, inputs=chat_inputs,
                   outputs=[chatbot, message])
    clear_button.click(clear_chat, inputs=None, outputs=[chatbot, message])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))