import os
from collections.abc import Generator

import gradio as gr
import openai

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_CLIENT = openai.Client(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

DEEPSEEK_GENERATE_KWARGS = {
    "temperature": max(float(os.getenv("TEMPERATURE", 0.9)), 1e-2),
    "max_tokens": int(os.getenv("MAX_NEW_TOKENS", 256)),
    "top_p": float(os.getenv("TOP_P", 0.6)),
    "frequency_penalty": max(-2, min(float(os.getenv("FREQ_PENALTY", 0)), 2)),
}


def query_deepseek(prompt: str) -> Generator[str, None, str]:
    messages = [{"role": "user", "content": prompt}]

    try:
        stream = DEEPSEEK_CLIENT.chat.completions.create(
            model="deepseek-v4-flash",
            messages=messages,
            **DEEPSEEK_GENERATE_KWARGS,
            stream=True,
        )
        output = ""
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                output += content
                yield output

    except Exception as e:
        if "Too Many Requests" in str(e):
            raise gr.Error("Too many requests on DeepSeek client") from e
        elif "You didn't provide an API key" in str(e):
            raise gr.Error("DeepSeek API key was either not provided or incorrect") from e
        else:
            raise gr.Error(f"Unhandled Exception: {str(e)}") from e
