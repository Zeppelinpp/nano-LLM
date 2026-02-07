from transformers import BertModel
from transformers import BertTokenizer


bert_model = "bert-base-uncased"
model = BertModel.from_pretrained(bert_model, cache_dir="../public")

tokenizer = BertTokenizer.from_pretrained(bert_model, cache_dir="../public")

sentence = "床前明"

input_ids = tokenizer.tokenize(sentence)
print(input_ids)
