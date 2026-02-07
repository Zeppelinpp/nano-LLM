import torch
import torch.nn as nn

input = torch.randn(3, 4)  # (seq, hidden)
w_q = nn.Linear(4, 4)
w_k = nn.Linear(4, 4)
w_v = nn.Linear(4, 4)

q, k, v = (
    w_q(input),
    w_k(input),
    w_v(input),
)

k_cache, v_cache = k, v

new_token = torch.randn(1, 4)
new_input = torch.cat([input, new_token], dim=0)
# print(new_input.shape) -> (4, 4)

new_q = w_q(new_token)
new_k_full = w_k(new_input)
new_k_use_cache = torch.cat([k_cache, w_k(new_token)], dim=0)
print(f"k_full: \n{new_k_full}\n")
print(f"k_use_cahe: \n{new_k_use_cache}\n")

if torch.allclose(new_k_full, new_k_use_cache):
    print("Identical")
