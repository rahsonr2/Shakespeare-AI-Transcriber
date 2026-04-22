import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

login("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")

@st.cache_resource
def load_model():
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    #model_id = "mistralai/Mistral-7B-Instruct-v0.2"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )

    return tokenizer, model


tokenizer, model = load_model()

def to_modern_english(text):
    prompt = f"""
You are a literary assistant.

Convert the following Shakespeare text into a clear modern English summary.

Rules:
- Keep meaning the same
- Do not add new ideas
- Be concise

TEXT:
{text}

MODERN ENGLISH SUMMARY:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.3,
        top_p=0.9
    )

    return tokenizer.decode(output[0], skip_special_tokens=True)

def to_shakespeare_style(summary, style_prompt):
    prompt = f"""
You are a Shakespearean language transformation engine.

Task:
Rewrite the modern English passage into Early Modern English (Shakespeare style).

Rules:
- Preserve meaning exactly
- Use archaic English (thou, thee, thy, art, doth)
- Do NOT add new events or ideas
- Make it poetic if possible

Style instruction:
{style_prompt}

TEXT:
{summary}

SHAKESPEARE STYLE OUTPUT:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=0.8,
        top_p=0.9
    )

    return tokenizer.decode(output[0], skip_special_tokens=True)

st.title("Hamlet to Modern English Translator")

st.write("Paste a Hamlet excerpt and transform it step-by-step into modern English and Shakespearean style.")

# INPUT TEXT
hamlet_text = st.text_area("Enter Hamlet Excerpt", height=250)

# STYLE INPUT
style_prompt = st.text_input(
    "Style Prompt",
    value="make it sound like Shakespeare"
)

# RUN BUTTON
if st.button("Run Transformation"):

    if not hamlet_text.strip():
        st.error("Please enter some text.")
    else:

        st.subheader("Step 1: Modern English Summary")

        modern = to_modern_english(hamlet_text)
        st.write(modern)

        st.subheader("Step 2: Shakespeare Style Rewrite")

        shakespeare = to_shakespeare_style(modern, style_prompt)
        st.write(shakespeare)

        output_text = f"""
========================
ORIGINAL TEXT
========================
{hamlet_text}

========================
MODERN ENGLISH SUMMARY
========================
{modern}

========================
SHAKESPEARE STYLE OUTPUT
========================
{shakespeare}
"""

        with open("output.txt", "w", encoding="utf-8") as f:
            f.write(output_text)

        st.success("Saved to output.txt")
        st.download_button(
            label="⬇ Download Output File",
            data=output_text,
            file_name="hamlet_transformation.txt",
            mime="text/plain"
        )