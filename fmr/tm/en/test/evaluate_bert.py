#!/usr/bin/env python
#SBATCH --time=8:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --mem=8GB
#SBATCH --gres=gpu:t4:1
# Gets bert predictions for each hole
# in each sentence in a document
# @author: Richard Yue

import sys
import fileinput
import gensim
from gensim.models import Word2Vec
from nltk.tokenize import RegexpTokenizer

from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path = "/work/nlp/yue.r/fuzzy_match_repair/fmr/tm/en/test"

model = AutoModelForMaskedLM.from_pretrained(f'{path}/distilbert-finetuned-europarl').to(device)
bert_tokenizer = AutoTokenizer.from_pretrained(f'{path}/distilbert-finetuned-europarl')

tokenizer = RegexpTokenizer(r'\w+')

def get_groups(holes:list):
    ''' Takes in a list of holes and returns groups of
        consecutive holes [(1, 2, 3), (5, 6)]
        @params: holes, the list of hole indices
        @returns: a list of group tuples
    '''
    groups = []
    start = 0
    end = 0
    for i in range(1, len(holes)):
        if holes[i] == holes[end] + 1:
            end = i
            if i == len(holes)-1:
                groups.append((holes[start], holes[end]))
        else:
            groups.append((holes[start], holes[end]))
            start = end = i
    return groups

def bert_predict(text:str, k=int):
    inputs = bert_tokenizer(text, return_tensors="pt").to(device)
    token_logits = model(**inputs).logits
    # Find the location of [MASK] and extract its logits
    mask_token_index = torch.where(inputs["input_ids"] == bert_tokenizer.mask_token_id)[1]
    mask_token_logits = token_logits[0, mask_token_index, :]
    # Pick the [MASK] candidates with the highest logits
    top_k_tokens = torch.topk(mask_token_logits, k, dim=1).indices[0].tolist()
    predictions = []
    for token in top_k_tokens:
        predictions.append(bert_tokenizer.decode([token]))
    return predictions

def predict_holes(sentence:str, holes:list, hole_groups:list):
    ''' Takes in a sentence and hole indices. Returns hole predictions.
    @params: sentence, the sentence to predict; holes, the indices of holes
    @returns: A list of tuples with the following format:
              ( 'hole index', 'predicted word', 'left/middle/right hole' )
              Note: left/middle/right hole indicates whether the hole
                    had left, middle, or right context.
                    For instance, in "The man _2_ _3_ _4_ store", 2 would get
                    left context (l), 3 would get left (l) as well,
                    4 would be both (lr) context.
    '''
    predictions = []
    sent_tokenized = tokenizer.tokenize(sentence.lower())
    prediction_groups = []
    """ This is for dealing with groups of holes
    for group in hole_groups:
        if group[0] == 0 and group[1] != len(sent_tokenized)-1:
            pred_sent = [sent_tokenized[group[0]], sent_tokenized[group[1]+1]]
        elif group[0] == 0:
            pred_sent = []
        elif group[1] == len(sent_tokenized)-1:
            pred_sent = [sent_tokenized[group[0]], sent_tokenized[group[1]]]
        else:
            pred_sent = [sent_tokenized[group[0]-1], sent_tokenized[group[1]+1]]
    """
    for hole in holes:
        if hole != 0 and hole != len(sent_tokenized)-1:
            pred_sent = f"{sent_tokenized[hole-1]} [MASK] {sent_tokenized[hole+1]}"
            predictions = bert_predict(pred_sent, k=50)
            if predictions != None:
                prediction_groups.append([key[0] for key in predictions])
    return prediction_groups

def main():
    lines = []
    for fileinput_line in fileinput.input(files=sys.argv[1]):
        if 'Exit' == fileinput_line.rstrip():
            break
        lines.append(fileinput_line.strip('\n'))
    for line in lines[1:]:
        split_line = line.split('\t')
        h = split_line[4]
        sentence = split_line[0]
        if len(h) > 0:
            holes_num = [int(hole) for hole in h.split(',')]
            holes = holes_num
            sentence_tokenized = tokenizer.tokenize(sentence.lower())
            holes_str = [sentence_tokenized[hole] for hole in holes]
            holes_str_set = set(holes_str)
            hole_groups = get_groups(holes)
            prediction_groups = predict_holes(sentence, holes, hole_groups)
            predictions = []
            sys.stdout.write(f'{sentence}\t')
            for item in holes_str_set:
                sys.stdout.write(f'{item},')
            sys.stdout.write('\t')
            for group in prediction_groups:
                holes_set = set(holes_str)
                group_set = set(group)
                intersect = holes_set.intersection(group_set)
                for item in intersect:
                    sys.stdout.write(f'{item},')
            sys.stdout.write('\n')
        if len(h) == 0:
            sys.stdout.write(f'{sentence}\tNone\tNone\n')
        
if __name__ == "__main__":
    main()
