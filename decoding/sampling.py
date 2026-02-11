from typing import Optional
import torch
from pydantic import BaseModel, Field
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

prompt = "The Captial of China is "
input_ids = tokenizer.encode(prompt, return_tensors="pt")


def generate(
    logits,
    strategy: str,
    temperature: Optional[float] = 1.0,
    top_p: Optional[float] = 0.0,
    top_k: Optional[int] = 0,
):
    logits = logits / temperature

    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        # torch.topk -> ([topk valuse], [topk indices]), [..., -1, None] -> Select the smallest value in topk and unsqueeze(-1)
        logits[indices_to_remove] = float("-inf")

    if top_p > 0.0:
        # Sort prob
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumsum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # torch.cumsum:
        # softmax(sorted logits) -> t1: prob_1, t2: prob_2, ...  sum(prob_i) = 1
        # cumsum_probs: 0: prob_1, 1: prob_1 + prob_2, 2: prob_1 + prob_2 + prob_3, ...

        sorted_indices_to_remove = cumsum_probs > top_p

        # Keep at least one token
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
            ..., :-1
        ].clone()  # [0, 1, 1] -> [?, 0, 1]
        sorted_indices_to_remove[..., 0] = 0  # [?, 0, 1] -> [0, 0, 1]

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = float("-inf")

    if strategy == "greedy":
        return torch.argmax(logits, dim=-1).unsqueeze(0)
    else:
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(
            probs, num_samples=1
        )  # argmax select the hightest prob everytime and multinomial select by prob
        return next_token


generated = input_ids
max_new_token = 20


class Config(BaseModel):
    strategy: str = Field(default="greedy")
    temperature: float = Field(default=1.0)
    top_k: int = Field(default=0)
    top_p: float = Field(default=0.0)


def generate_seq(max_new_token: int, input_ids, config: Config):
    for _ in range(max_new_token):
        with torch.no_grad():
            outputs = model(input_ids)
            next_token_logits = outputs.logits[0, -1, :]

            next_token = generate(
                next_token_logits,
                strategy=config.strategy,
                temperature=config.temperature,
                top_k=config.top_k,
                top_p=config.top_p,
            )

            input_ids = torch.cat((input_ids, next_token.unsqueeze(0)), dim=1)

    print(f"Result with strategy {config.strategy.upper()}\n")
    print(f"- Temperature: {config.temperature}")
    print(f"- Top K: {config.top_k}")
    print(f"- Top P: {config.top_p}")

    print(f"- Generated: {tokenizer.decode(input_ids[0])}")


config = Config(strategy="greedy", temperature=1.0, top_k=0, top_p=0.0)
generate_seq(max_new_token, input_ids, config)
config = Config(strategy="temperature", temperature=0.7, top_k=0, top_p=0.0)
generate_seq(max_new_token, input_ids, config)
config = Config(strategy="top_k", top_k=10, temperature=1.0, top_p=0.0)
generate_seq(max_new_token, input_ids, config)
config = Config(strategy="top_p", top_p=0.9, temperature=1.0, top_k=0)
generate_seq(max_new_token, input_ids, config)

