#!/usr/bin/env python
# coding: utf-8

# In[1]:


from transformers import AutoTokenizer
from transformers import AutoModelForMaskedLM
from datasets import load_dataset

model = AutoModelForMaskedLM.from_pretrained('distilbert-base-uncased')
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')


# In[3]:


text = "This is a great [MASK]"


# In[4]:


import torch

inputs = tokenizer(text, return_tensors="pt")
token_logits = model(**inputs).logits

mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
mask_token_logits = token_logits[0, mask_token_index, :]
top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()

for token in top_5_tokens:
    print(f"'>>> {text.replace(tokenizer.mask_token, tokenizer.decode([token]))}'")


# In[17]:


dataset = load_dataset("text", data_files={"train": "../train/train", "test": "test", "dev":"../dev/dev"})
dataset


# In[18]:


def tokenize_function(examples):
    result = tokenizer(examples["text"], max_length=128, padding='max_length', truncation=True)
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result

tokenized_datasets = dataset.map(
    tokenize_function, batched=True
)
tokenized_datasets


# In[19]:


labels = tokenized_datasets["train"]["input_ids"]
labeled_dataset_train = tokenized_datasets["train"].add_column("labels", labels)
labeled_dataset_train


# In[20]:


labels = tokenized_datasets["dev"]["input_ids"]
labeled_dataset_dev = tokenized_datasets["dev"].add_column("labels", labels)
labeled_dataset_dev


# In[21]:


import collections
import numpy as np

from transformers import default_data_collator

wwm_probability = 0.2


def whole_word_masking_data_collator(features):
    for feature in features:
        word_ids = feature.pop("word_ids")

        # Create a map between words and corresponding token indices
        mapping = collections.defaultdict(list)
        current_word_index = -1
        current_word = None
        for idx, word_id in enumerate(word_ids):
            if word_id is not None:
                if word_id != current_word:
                    current_word = word_id
                    current_word_index += 1
                mapping[current_word_index].append(idx)

        # Randomly mask words
        mask = np.random.binomial(1, wwm_probability, (len(mapping),))
        input_ids = feature["input_ids"]
        labels = feature["labels"]
        new_labels = [-100] * len(labels)
        for word_id in np.where(mask)[0]:
            word_id = word_id.item()
            for idx in mapping[word_id]:
                new_labels[idx] = labels[idx]
                input_ids[idx] = tokenizer.mask_token_id
        feature["labels"] = new_labels

    return default_data_collator(features)


# In[22]:


samples = [labeled_dataset_train[i] for i in range(2)]
batch = whole_word_masking_data_collator(samples)

for chunk in batch["input_ids"]:
    print(f"\n'>>> {tokenizer.decode(chunk)}'")


# In[28]:


from transformers import TrainingArguments

batch_size = 64

logging_steps = len(labeled_dataset_train)
model_name = "distilbert-finetuned-europarl"

training_args = TrainingArguments(
    output_dir=model_name,
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    num_train_epochs=10,
    learning_rate=2e-5,
    weight_decay=0.01,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    push_to_hub=False,
    fp16=True,
    logging_steps=logging_steps,
    remove_unused_columns=False
)


# In[29]:


from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=labeled_dataset_train,
    eval_dataset=labeled_dataset_dev,
    data_collator=whole_word_masking_data_collator,
    tokenizer=tokenizer,
)


# In[30]:


print(len(labeled_dataset_train[9]['input_ids']))


# In[31]:


import math

eval_results = trainer.evaluate()
print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")


# In[32]:


trainer.train()


# In[ ]:


get_ipython().system('jupyter nbconvert --to script be.ipynb')

