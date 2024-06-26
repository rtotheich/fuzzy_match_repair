#!/usr/bin/env python

#SBATCH --partition=short
#SBATCH --time=1409

# Get sentences with hole included and
# print their accuracy in a CSV format

from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')

import sys
import fileinput
sys.path.append('/work/nlp/yue.r/fuzzy_match_repair/fmr/tm/en/test/')
from evaluate_w2v_google import predict_holes
from evaluate_bert import bert_predict

def main():
    k = int(sys.argv[2])
    lines = []
    for line in fileinput.input(sys.argv[1]):
        if 'Exit' == line.rstrip():
            break
        lines.append(line)
    for line in lines[1:]:
        split_line = line.strip('\n').split('\t')
        holes = split_line[4].split(',')
        if holes[0] != '':
            int_holes = [int(hole) for hole in holes]
            holes = int_holes
            sentence = split_line[0]
            tok_sentence = tokenizer.tokenize(sentence.lower())
            for hole in holes:
                if hole != 0 and hole != len(tok_sentence)-1:
                    tri_gram = f'{tok_sentence[hole-1]} {tok_sentence[hole]} {tok_sentence[hole+1]}'
                    bert_mask_tri_gram = f'{tok_sentence[hole-1]} [MASK] {tok_sentence[hole+1]}'
                    bert_predictions = ','.join(bert_predict(bert_mask_tri_gram, k)[:k])
                    w2v_predictions = ','.join(predict_holes(sentence, holes, [])[0][:k])
                    sys.stdout.write(f'{tri_gram}\t{bert_predictions}\t{w2v_predictions}\n')
                
    
if __name__ == '__main__':
    main()
