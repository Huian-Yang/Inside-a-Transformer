import time
import torch
import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from utils import top_k_top_p_filtering, create_probability_table



#generate a token from input using set hyperparameters (greedy/sampling, temperature,
#top-k, top-p). 
#returns the top tokens for display.
def generate_token_step(
    model, 
    tokenizer, 
    input_ids: torch.Tensor, 
    decoding_strategy: str = "Greedy",
    temperature: float = 1.0,
    top_k: int = 6,
    top_p: float = 1.0,
    display_top: int = 6,
    skip_special_tokens: bool = True
):

    with torch.no_grad():
        outputs = model(input_ids)
    logits = outputs.logits[:, -1, :]
    logits = logits / max(temperature, 1e-9)

    if decoding_strategy == "Greedy":
        probs = torch.softmax(logits, dim=-1)
        chosen_index = torch.argmax(probs, dim=-1).item()
        top_probs_tensor, top_idx_tensor = torch.topk(probs, display_top)
        top_probs = top_probs_tensor[0].tolist()
        top_indices = top_idx_tensor[0].tolist()
    else:
        filtered_logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
        filtered_probs = torch.softmax(filtered_logits, dim=-1)
        chosen_index = torch.multinomial(filtered_probs, 1).item()

        probs_for_display = torch.softmax(logits, dim=-1)
        top_probs_tensor, top_idx_tensor = torch.topk(probs_for_display, display_top)
        top_probs = top_probs_tensor[0].tolist()
        top_indices = top_idx_tensor[0].tolist()

    token_text = tokenizer.decode([chosen_index], skip_special_tokens=skip_special_tokens)
    top_tokens = [tokenizer.decode([idx], skip_special_tokens=skip_special_tokens) for idx in top_indices]

    return chosen_index, token_text, top_probs, top_tokens


#generating response -> showing each step and each word generated (updating placeholder).
#generation stops if reset button is clicked.
def generate_and_visualize_step_by_step(
    prompt: str,
    model_name: str = "gpt2",
    decoding_strategy: str = "Greedy",
    temperature: float = 1.0,
    top_k_val: int = 6,
    top_p_val: float = 1.0,
    max_new_tokens: int = 20,
    stop_on_punctuation: bool = True,
    sleep_time: float = 0.5
):
   
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    prompt_ids = tokenizer(prompt + "\n", return_tensors="pt").input_ids
    input_ids = prompt_ids.clone()
    generated_ids = []
    probabilities_data = []

    partial_text_placeholder = st.empty()
    partial_table_placeholder = st.empty()

    for step in range(max_new_tokens):
        if st.session_state.stop_generation:
            st.write("Generation stopped by user.")
            break

        chosen_index, token_text, top_probs, top_tokens = generate_token_step(
            model=model,
            tokenizer=tokenizer,
            input_ids=input_ids,
            decoding_strategy=decoding_strategy,
            temperature=temperature,
            top_k=top_k_val,
            top_p=top_p_val,
            display_top=6
        )

        if step == 0:
            probabilities_data.append((f"[SKIPPED: {token_text}]", top_probs, top_tokens))
        else:
            generated_ids.append(chosen_index)
            probabilities_data.append((token_text, top_probs, top_tokens))

        partial_generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        partial_text_placeholder.markdown(f"**Partial Generated Text**: {partial_generated_text}")

        #update the probs table
        step_df = create_probability_table(probabilities_data[-1:], start_index=len(probabilities_data))
        partial_table_placeholder.dataframe(step_df)

        input_ids = torch.cat([input_ids, torch.tensor([[chosen_index]])], dim=1)

        if stop_on_punctuation and token_text.strip() in {".", "?", "!"}:
            break

        time.sleep(sleep_time)

    partial_text_placeholder.empty()
    partial_table_placeholder.empty()

    full_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return full_text, probabilities_data
