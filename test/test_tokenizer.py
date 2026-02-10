from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
print(f"Tokenizer's vocab size: {tokenizer.vocab_size}")
print(f"Tokenizer's Padding token: {tokenizer.eos_token}")

tokenizer.pad_token = tokenizer.eos_token
max_length = 10
sequence = [
    "Hallo, Wo ist der Imbiss?",
    "Meine wurst ist gut.",
    "Bis du hungrig?",
    "Das Salat ist billig",
]

result = tokenizer(sequence, truncation=True, padding=True, max_length=max_length)
print(result)

# input_ids = [34194, 78, 11]
# decode_result = tokenizer.decode(input_ids, attention_mask=[1, 1, 1])
# print(decode_result)
