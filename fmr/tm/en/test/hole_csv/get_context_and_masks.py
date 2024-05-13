import sys

from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')

with open(sys.argv[1], 'r') as fd:
    lines = fd.readlines()

for line in lines:
    split_line = line.split('\t')
    sentence = split_line[0]
    hole_nums = split_line[4]
    for hole in hole_nums.split(','):
        if hole != 'None' and hole != '\n' and int(hole) != 0:
            sent_tokenized = tokenizer.tokenize(sentence.lower())
            if int(hole) != len(sent_tokenized)-1:
                print(sent_tokenized[int(hole)-1], sent_tokenized[int(hole)], sent_tokenized[int(hole)+1])
    
