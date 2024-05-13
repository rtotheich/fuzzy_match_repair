# Run inference on BERT

from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer
import torch

print('done')

ckpt = 'distilbert-finetuned-europarl/checkpoint-43000'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = AutoModelForMaskedLM.from_pretrained(ckpt).to(device)
tokenizer = AutoTokenizer.from_pretrained(ckpt).to(device)

def predict(text:str):
    inputs = tokenizer(text, return_tensors="pt")
    token_logits = model(**inputs).logits
    # Find the location of [MASK] and extract its logits
    mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
    mask_token_logits = token_logits[0, mask_token_index, :]
    # Pick the [MASK] candidates with the highest logits
    top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()

    for token in top_5_tokens:
        print(f"'>>> {text.replace(tokenizer.mask_token, tokenizer.decode([token]))}'")

inp = ''
while inp != 'quit':
    inp = input('Enter text:\n')
    if inp != 'quit':
        predict(inp)
