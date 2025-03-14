import streamlit as st
import pandas as pd
from generation import generate_and_visualize_step_by_step
from utils import create_probability_table

st.title("Inside a Transformer")


#reset
if "stop_generation" not in st.session_state:
    st.session_state.stop_generation = False

def stop_generation_callback():
    st.session_state.stop_generation = True

stop_button = st.button("Reset", on_click=stop_generation_callback)


#prompt Input
prompt = st.text_input("Enter a prompt to generate from:")

st.markdown("## Decoding Strategy")
st.markdown(
    """<small>
    <strong>Greedy:</strong> The model consistently picks the single most likely next token at each step, leading to a more predictable and deterministic output.<br/>
    <strong>Sampling:</strong> The model randomly selects from the probability distribution of possible tokens, introducing spontaneity and creative variety into the generated text.
    </small>""",
    unsafe_allow_html=True
)
decoding_strategy = st.radio(
    "",
    ["Greedy", "Sampling"],
    index=0
)


#temperature
st.markdown("## Temperature")
st.markdown(
    """<small>
    Scales the logits before softmax.<br/>
    <strong>Lower (&lt;1):</strong> Makes distribution sharper, more deterministic.<br/>
    <strong>Higher (&gt;1):</strong> Flattens distribution, increasing randomness.
    </small>""",
    unsafe_allow_html=True
)
temperature = st.slider(
    "",
    0.0, 2.0, 1.0, 0.1
)


#top-k
st.markdown("## Top-k")
st.markdown(
    """<small>
    Restricts token selection to the top k most likely tokens.<br/>
    Higher values allow more diverse options, lower values focus on the most probable tokens.
    </small>""",
    unsafe_allow_html=True
)
top_k_val = st.slider(
    "",
    min_value=1,
    max_value=50,
    value=6,
    step=1
)


#top-p
st.markdown("## Top-p (Nucleus Sampling)")
st.markdown(
    """<small>
    Chooses tokens from the smallest set whose cumulative probability reaches p.<br/>
    <strong>Lower (&lt;1):</strong> Limits the sampling to higher-probability tokens.<br/>
    <strong>1.0:</strong> Disables nucleus filtering, allowing full distribution.
    </small>""",
    unsafe_allow_html=True
)
top_p_val = st.slider(
    "",
    min_value=0.0,
    max_value=1.0,
    value=1.0,
    step=0.05
)


#max New Tokens
st.markdown("## Max New Tokens")
st.markdown(
    """<small>
    Controls the maximum number of tokens to generate after the prompt.<br/>
    Helps limit or extend the length of generated text.
    </small>""",
    unsafe_allow_html=True
)
max_new_tokens_val = st.slider(
    "",
    min_value=1,
    max_value=100,
    value=20,
    step=1
)


#stop on Punctuation
st.markdown("## Stop on Punctuation")
st.markdown(
    """<small>
    When checked, generation stops upon encountering '.', '?', or '!'.<br/>
    Ensures more complete sentences or shorter outputs.
    </small>""",
    unsafe_allow_html=True
)
stop_on_punc = st.checkbox("Enable punctuation stop", value=True)


# Model Selection
model_options = ["gpt2", "gpt2-medium", "gpt2-large"]
selected_models = st.multiselect(
    "Select models to compare",
    options=model_options,
    default=["gpt2"]
)

if st.button("Generate"):
    st.session_state.stop_generation = False

    if prompt and len(selected_models) > 0:
        for model_name in selected_models:
            st.markdown(
                f"## Model: <span style='color: #00cc99;'>{model_name}</span>", 
                unsafe_allow_html=True
            )
            with st.spinner(f"Generating text step by step with {model_name}..."):
                final_text, probs_data = generate_and_visualize_step_by_step(
                    prompt=prompt,
                    model_name=model_name,
                    decoding_strategy=decoding_strategy,
                    temperature=temperature,
                    top_k_val=top_k_val,
                    top_p_val=top_p_val,
                    max_new_tokens=max_new_tokens_val,
                    stop_on_punctuation=stop_on_punc,
                    sleep_time=0.5
                )
            
            st.write("### Final Generated Text:")
            st.write(final_text)

            if probs_data:
                st.write("### Full Probability Table:")
                full_table = create_probability_table(probs_data, start_index=1)
                st.dataframe(full_table)
            
            st.write("---")
