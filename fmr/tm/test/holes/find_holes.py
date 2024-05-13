# Program to find the holes in fuzzy match segments
# @author: Richard Yue

import sys, fileinput
sys.path.append('/work/nlp/yue.r/fuzzy_match_repair/fmr')
from diff_lib import string_diff

def main():
    if len(sys.argv) < 2:
        print('Please input a file as command line argument')
        return
    print('segment\tbest\tscore\ttokens\indices')
    for fileinput_line in fileinput.input(sys.argv[1]):
        if 'Exit' == fileinput_line.rstrip():
            break
        line = fileinput_line.strip('\n')
        split_line = line.split('\t')
        holes = string_diff(split_line[0], split_line[1])
        sys.stdout.write(f"{line}\t{holes[0]}\t{holes[1]}\n")

if __name__ == '__main__':
    main()
