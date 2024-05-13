# Get difference between two strings
# @author: Richard Yue

import difflib
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')

def remove_chars(s:str, chars:list):
    for char in chars:
        s = s.replace(char, '')
    return s

def string_diff(s1, s2):
    # s1 = remove_chars(s1, [',', '.', '!', '"', "'", '“', '”', '’', '`', '(', ')', '?', ':', ';'])
    # s2 = remove_chars(s2, [',', '.', '!', '"', "'", '“', '”', '’', '`', '(', ')', '?', ':', ';'])
    # s1 = ' '.join(s1.split('-')).split()
    # s2 = ' '.join(s2.split('-')).split()
    s2 = tokenizer.tokenize(s2.lower())
    s1 = tokenizer.tokenize(s1.lower())
    holes = []
    tokens = []
    differences = []
    offset = 0
    bad_starts = ['?', '-']
    diff_out = list(difflib.ndiff(s2, s1))
    clean_diff = []
    count = 0
    for item in diff_out:
        if item[0] not in bad_starts:
            clean_diff.append(item)
    for i,diff in enumerate(clean_diff):
        if diff[0] == '+':
            tokens.append(diff[2:])
            differences.append(i)
    tokens_string = ','.join(tokens)
    differences_string = ','.join([str(item) for item in differences])
    return (tokens_string, differences_string)

def main():
    s1 = "[2100 – 2200]"
    s2 = "[21000 - 22000]"
    print(f"Example new segment: {s1}")
    print(f"Example tm segment: {s2}")
    token_diff = string_diff(s1, s2)
    print(f"Tokens: {token_diff[0]}\nDifferences: {token_diff[1]}")

if __name__ == "__main__":
    main()
