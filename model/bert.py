from transformers import BertModel, BertTokenizer


# 使用中文 BERT 模型
bert_model = "bert-base-chinese"
model = BertModel.from_pretrained(bert_model, cache_dir="../public")
tokenizer = BertTokenizer.from_pretrained(bert_model, cache_dir="../public")

sentence = "床前明月光"
tokens = tokenizer.tokenize(sentence)
print(tokens)

# 如果需要获取 token IDs
input_ids = tokenizer.encode(sentence, add_special_tokens=True)
print("Input IDs:", input_ids)
print("Decoded:", tokenizer.decode(input_ids))
