import torch
import pandas as pd

#Filter a distribution of logits using top-k and/or nucleus (top-p) filtering.
def top_k_top_p_filtering(
    logits: torch.Tensor, 
    top_k: int = 50, 
    top_p: float = 1.0
) -> torch.Tensor:
    top_k = max(top_k, 1) 
    if top_k < logits.size(-1):
        values_to_keep, _ = torch.topk(logits, top_k)
        min_threshold = values_to_keep[..., -1, None]
        logits = torch.where(logits < min_threshold, torch.tensor(float('-inf'), device=logits.device), logits)
    if 0.0 <= top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        cutoff = (cumulative_probs > top_p).float().cumsum(dim=-1)
        keep_indices = cutoff < 1
        sorted_logits = sorted_logits.masked_fill(~keep_indices, float('-inf'))
        logits_ = torch.zeros_like(logits) + float('-inf')
        logits_.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
        logits = logits_
    return logits


#build dataframe from a list of (token_text, [probs], [tokens]).
def create_probability_table(prob_data, start_index=1):
    columns = [f"Word {start_index + i}" for i in range(len(prob_data))]
    rows = ["Generated Word"] + [f"Choice {i+1}" for i in range(5)]
    table_data = []
    for row_idx in range(6):
        row = []
        for (token_text, probs, token_choices) in prob_data:
            if row_idx == 0:
                row.append(token_text)
            else:
                alt_text = token_choices[row_idx - 1]
                alt_prob = probs[row_idx - 1]
                row.append(f"{alt_text} ({alt_prob:.2%})")
        table_data.append(row)
    return pd.DataFrame(table_data, index=rows, columns=columns)
