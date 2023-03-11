# https://qiita.com/kenta1984/items/7f3a5d859a15b20657f3

import torch
import sys
from transformers import BertJapaneseTokenizer, BertModel

print(torch.cuda.is_available())

# load markdown files
with open("./documents/2_2_release_notes.md", "r") as f:
    lines = f.readlines()

# Load pre-trained tokenizer
tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')

# Load pre-trained model
model = BertModel.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
model.to(device)
model.eval()

def t2v(model, tokenizer, text):
    tokens = tokenizer(text)["input_ids"]
    inputs = torch.tensor(tokens, device=device).reshape(1, -1)
    with torch.no_grad():
        outputs = model(inputs, output_hidden_states=True)
        last_hidden_state = outputs.last_hidden_state[0]
        averaged_hidden_state = last_hidden_state.sum(dim=0) / len(last_hidden_state)
    return averaged_hidden_state

def search_similarity(query_embedding, lines, embeddings):
    similarities = []
    for line, embedding in zip(lines, embeddings):
        similarity = torch.nn.functional.cosine_similarity(query_embedding, embedding, dim=0)
        similarities.append([similarity, line]) 
    return sorted(similarities)
    #return sorted(similarities, reverse=True)

embeddings = []
for line in lines:
    embeddings.append(t2v(model, tokenizer, line))


while True:
    print("plese input sentence!")
    #s = input()
    s = "キャッシュを効かせる"
    query = t2v(model, tokenizer, s)
    results = search_similarity(query, lines, embeddings)
    for result in results:
        print(result)
    break
