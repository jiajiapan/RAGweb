import os
from typing import Any, Dict, Generator, List
import gradio as gr
import openai

KIMI_KEY = os.getenv("KIMI_API_KEY")
KIMI_CLIENT= openai.Client(api_key = KIMI_KEY, base_url = "https://api.deepseek.com")

KIMI_GENERATE_KWARGS = {
    "temperature": max(float(os.getenv("TEMPERATURE", 0.9)), 1e-2),
    "max_tokens": int(os.getenv("MAX_NEW_TOKENS", 256)),
    "top_p": float(os.getenv("TOP_P", 0.6)),
    "frequency_penalty": max(-2, min(float(os.getenv("FREQ_PENALTY", 0)), 2)),
}

def query_kimi(prompt: str) -> Generator[str, None, str]:
    messages = [{"role": "user", "content": prompt}]

    try:
        stream = KIMI_CLIENT.chat.completions.create(
            model = "deepseek-v4-flash",
            messages = messages,
            **KIMI_GENERATE_KWARGS,
            stream = True,
        )
        output = ""
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                output += content
                yield output
    
    except Exception as e:
        if "Too Many Requests" in str(e):
            raise gr.Error("Too many requests on Kimi client")
        elif "You didn't provide an API key" in str(e):
            raise gr.Error("Kimi key was either not provided or incorrect")
        else:
            raise gr.Error(f"Unhandled Exception: {str(e)}")