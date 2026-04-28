import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from sentence_transformers import SentenceTransformer, util

login("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")

@st.cache_resource
def load_model():
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    return tokenizer, model

tokenizer, model = load_model()

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

def compute_bert_similarity(text1, text2):
    embeddings = embedder.encode([text1, text2], convert_to_tensor=True)
    similarity = util.cos_sim(embeddings[0], embeddings[1])
    return similarity.item()

def get_similarity_label(score):
    if score >= 0.75:
        return "High"
    elif score >= 0.5:
        return "Medium"
    else:
        return "Low"

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
        temperature=0.4,
        top_p=0.9
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

def to_shakespeare_style(summary, style_prompt):
    prompt = f"""
You are a Shakespearean rewriting engine.

You MUST rewrite the text into Early Modern English.

STRICT RULES:
- You are NOT allowed to copy sentences from the input
- You MUST change sentence structure
- You MUST use archaic words (thou, thee, thy, hath, doth, art)
- You MUST make it poetic and dramatic
- Keep the same meaning, but fully rewrite

Style instruction:
{style_prompt}

TEXT:
{summary}

REWRITTEN SHAKESPEAREAN VERSION:
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=1.0,
        top_p=0.95,
        repetition_penalty=1.2
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

def clear_fields():
    st.session_state.hamlet_text = ""
    st.session_state.selected_style = "Select..."
    st.session_state.custom_style = ""

st.title("Hamlet to Modern English Translator")

style_options = [
    "Select...",
    "Make it sound like Shakespeare",
    "Darker and more tragic tone",
    "Highly poetic with rich metaphors",
    "Short and dramatic dialogue style",
    "Formal royal speech tone",
    "Other"
]

hamlet_text = st.text_area(
    "Enter Hamlet Excerpt",
    height=250,
    key="hamlet_text"
)

selected_style = st.selectbox(
    "Choose a Style",
    style_options,
    key="selected_style"
)

custom_style = ""
if selected_style == "Other":
    custom_style = st.text_input(
        "Enter your custom style",
        key="custom_style"
    )

final_style_prompt = (
    st.session_state.custom_style.strip()
    if selected_style == "Other"
    else selected_style
)

if st.button("Run Transformation"):

    if not hamlet_text.strip():
        st.error("Please enter some text.")
    elif selected_style == "Select...":
        st.error("Please choose a style.")
    elif selected_style == "Other" and not st.session_state.custom_style.strip():
        st.error("Please enter a custom style prompt.")
    else:

        st.subheader("Step 1: Modern English Summary")
        modern = to_modern_english(hamlet_text)
        st.write(modern)

        cosine1 = compute_bert_similarity(hamlet_text, modern)
        label1 = get_similarity_label(cosine1)

        st.subheader("Step 1 Similarity (Original vs Modern)")
        st.success(f"Similarity: {cosine1:.4f} ({label1})")
        st.metric("Step 1 Similarity", f"{cosine1:.4f}", label1)

        st.subheader("Step 2: Shakespeare Style Rewrite")
        shakespeare = to_shakespeare_style(modern, final_style_prompt)
        st.write(shakespeare)

        cosine2 = compute_bert_similarity(modern, shakespeare)
        label2 = get_similarity_label(cosine2)

        st.subheader("Step 2 Similarity (Modern vs Shakespeare)")
        st.success(f"Similarity: {cosine2:.4f} ({label2})")
        st.metric("Step 2 Similarity", f"{cosine2:.4f}", label2)

        output_text = f"""
ORIGINAL TEXT
========================
{hamlet_text}

MODERN ENGLISH SUMMARY
========================
{modern}

SHAKESPEARE STYLE OUTPUT
========================
{shakespeare}

STEP 1 SIMILARITY (ORIGINAL vs MODERN)
========================
Score: {cosine1:.4f}
Level: {label1}

STEP 2 SIMILARITY (MODERN vs SHAKESPEARE)
========================
Score: {cosine2:.4f}
Level: {label2}

STYLE USED
========================
{final_style_prompt}
"""

        with open("output.txt", "w", encoding="utf-8") as f:
            f.write(output_text)

        st.success("Saved to output.txt")

        st.download_button(
            label="Download Output File",
            data=output_text,
            file_name="hamlet_transformation.txt",
            mime="text/plain"
        )

        st.button("Clear", on_click=clear_fields)
