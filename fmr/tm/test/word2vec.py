#!/usr/bin/env python
# word2vec prediction
import gensim
from gensim.models import Word2Vec 
from gensim.models import KeyedVectors
import pandas as pd
from nltk.tokenize import RegexpTokenizer

with open('test', 'r') as file:
    text = file.readlines()

string = "Article 8 (7) of Regulation (EU) No 1321/2014 is identical to Article 3 (7) of that Regulation.\tAnnex I to Regulation (EU) No 208/2014 is amended as set out in the Annex to this Regulation.\t66\tArticle,8,7,of,1321/2014,identical,Article,3,7,of,that\t0,1,2,3,7,9,11,12,13,14,15"

split_string = string.split('\t')
holes = split_string[4].split(',')
seg1 = split_string[0].split()
for hole in enumerate(holes):
    seg1.pop(int(hole[0]-int(hole[1])))

tokenizer = RegexpTokenizer(r'\w+')
sentences_tokenized = [w.lower() for w in sentences]
sentences_tokenized = [tokenizer.tokenize(i) for i in sentences_tokenized]

# model.save("word2vec.model")

# encoded_seg = [word.encode('utf-8') for word in seg1]
# print(model.predict_output_word(encoded_seg))
# print(split_string[3])
