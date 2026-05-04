import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer, util

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

@st.cache_resource
def load_embedder():
    return SentenceTransformer("BAAI/bge-large-en-v1.5")

embedder = load_embedder()

MODEL_NAME = "llama-3.3-70b-versatile"

def query_model(prompt, max_tokens=200, temperature=0.5):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.9
    )
    return response.choices[0].message.content.strip()

def stream_model(prompt, max_tokens=200, temperature=0.5):
    stream = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.9,
        stream=True
    )

    full_text = ""
    placeholder = st.empty()

    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta and delta.content:
            full_text += delta.content
            placeholder.markdown(full_text)

    return full_text.strip()

def compute_bert_similarity(text1, text2):
    embeddings = embedder.encode([text1, text2], convert_to_tensor=True)
    similarity = util.cos_sim(embeddings[0], embeddings[1])
    return float(similarity)

def get_similarity_label(score):
    if score >= 0.75:
        return "High"
    elif score >= 0.5:
        return "Medium"
    return "Low"

def to_modern_english_stream(text):
    prompt = f"""
Convert this Shakespeare text into clear modern English.

Rules:
- Keep meaning the same
- Be concise
- No extra commentary

TEXT:
{text}
"""
    return stream_model(prompt, max_tokens=150, temperature=0.3)

def to_shakespeare_stream(summary, style_prompt):
    prompt = f"""
Rewrite this into Shakespearean English.

Rules:
- Do NOT copy wording
- Use archaic language (thou, thee, thy, hath, doth)
- Make it poetic
- Keep meaning

STYLE:
{style_prompt}

TEXT:
{summary}
"""
    return stream_model(prompt, max_tokens=250, temperature=0.9)

def query_shakespeare(summary, style_prompt):
    prompt = f"""
Rewrite this into Shakespearean English.

Rules:
- Do NOT copy wording
- Use archaic language (thou, thee, thy, hath, doth)
- Make it poetic
- Keep meaning

STYLE:
{style_prompt}

TEXT:
{summary}
"""
    return query_model(prompt, max_tokens=250, temperature=0.9)

def critic_feedback(generated_text, score):
    prompt = f"""
You are an English scholar critiquing Shakespeare-style writing.

This generated text scored {score:.4f} similarity.

TEXT:
{generated_text}

Explain:
- Why it is not fully Shakespearean
- What is missing
- Specific improvements

Only give feedback.
"""
    return query_model(prompt, max_tokens=150, temperature=0.6)

def improve_prompt(old_prompt, feedback):
    prompt = f"""
You are refining a prompt to better mimic Shakespeare.

ORIGINAL PROMPT:
{old_prompt}

FEEDBACK:
{feedback}

Return ONLY an improved prompt.
"""
    return query_model(prompt, max_tokens=100, temperature=0.7)

def run_agentic_pipeline(modern, initial_prompt):
    current_prompt = initial_prompt
    best_score = 0
    best_output = ""
    best_prompt = current_prompt

    history = []

    for i in range(3):
        shakespeare = query_shakespeare(modern, current_prompt)
        score = compute_bert_similarity(modern, shakespeare)

        feedback = critic_feedback(shakespeare, score)
        new_prompt = improve_prompt(current_prompt, feedback)

        history.append({
            "iteration": i + 1,
            "prompt": current_prompt,
            "output": shakespeare,
            "score": score,
            "feedback": feedback,
            "new_prompt": new_prompt
        })

        if score > best_score:
            best_score = score
            best_output = shakespeare
            best_prompt = current_prompt

        current_prompt = new_prompt

    return best_output, best_score, best_prompt, history

def clear_fields():
    st.session_state.hamlet_text = ""
    st.session_state.selected_style = "Select..."
    st.session_state.custom_style = ""

st.title("Hamlet to Modern English Transcriber")

style_options = [
    "Select...",
    "Make it sound like Shakespeare",
    "Darker and more tragic tone",
    "Highly poetic with rich metaphors",
    "Short and dramatic dialogue style",
    "Formal royal speech tone",
    "Other"
]

hamlet_text = st.text_area("Enter Hamlet Excerpt", height=250, key="hamlet_text")
selected_style = st.selectbox("Choose a Style", style_options, key="selected_style")

custom_style = st.session_state.get("custom_style", "")

if selected_style == "Other":
    st.text_input("Enter your custom style", key="custom_style")
    custom_style = st.session_state.get("custom_style", "")

final_style_prompt = custom_style.strip() if selected_style == "Other" else selected_style

col1, col2 = st.columns(2)
run_default = col1.button("Run Default")
run_agentic = col2.button("Run Agentic")

if run_default or run_agentic:

    if not hamlet_text.strip():
        st.error("Please enter some text.")
        st.stop()

    if selected_style == "Select...":
        st.error("Please choose a style.")
        st.stop()

    if selected_style == "Other" and not custom_style.strip():
        st.error("Please enter a custom style prompt.")
        st.stop()

    st.subheader("Step 1: Modern English Summary")
    modern = to_modern_english_stream(hamlet_text)

    cosine1 = compute_bert_similarity(hamlet_text, modern)
    label1 = get_similarity_label(cosine1)

    st.subheader("Step 1 Similarity")
    st.success(f"{cosine1:.4f} ({label1})")

    st.subheader("Step 2: Shakespeare Style Rewrite")

    agent_history_text = ""

    if run_default:
        shakespeare = to_shakespeare_stream(modern, final_style_prompt)
        cosine2 = compute_bert_similarity(modern, shakespeare)

    else:
        shakespeare, cosine2, final_prompt_used, history = run_agentic_pipeline(
            modern,
            final_style_prompt
        )

        st.subheader("Agentic Improvement Process")

        for step in history:
            st.markdown(f"Iteration {step['iteration']}")
            st.markdown("Prompt Used")
            st.write(step["prompt"])
            st.markdown("Generated Output")
            st.write(step["output"])
            st.markdown("Score")
            st.success(f"{step['score']:.4f}")
            st.markdown("Critique")
            st.info(step["feedback"])
            st.markdown("Improved Prompt")
            st.code(step["new_prompt"])

            agent_history_text += f"""
ITERATION {step['iteration']}
========================
PROMPT USED
{step['prompt']}

OUTPUT
{step['output']}

SCORE
{step['score']:.4f}

CRITIQUE
{step['feedback']}

IMPROVED PROMPT
{step['new_prompt']}

"""

        st.subheader("Final Output")
        st.markdown(shakespeare)

    label2 = get_similarity_label(cosine2)

    st.subheader("Step 2 Similarity")
    st.success(f"{cosine2:.4f} ({label2})")

    if run_agentic:
        st.subheader("Final Prompt Used")
        st.write(final_prompt_used)

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

STEP 1 SIMILARITY
========================
Score: {cosine1:.4f}
Level: {label1}

STEP 2 SIMILARITY
========================
Score: {cosine2:.4f}
Level: {label2}
"""

    if run_agentic:
        output_text += f"""

FINAL PROMPT USED
========================
{final_prompt_used}

AGENTIC PROCESS
========================
{agent_history_text}
"""

    st.download_button(
        label="Download Output File",
        data=output_text,
        file_name="hamlet_transcribed.txt",
        mime="text/plain"
    )

    st.button("Clear", on_click=clear_fields)
