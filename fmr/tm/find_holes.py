# Program to find the holes in fuzzy match segments
# @author: Richard Yue

import sys, fileinput
sys.path.append('/work/nlp/yue.r/fuzzy_match_repair/fmr')
from diff_lib import string_diff

def main():
    if len(sys.argv) < 2:
        print('Please input a file as command line argument')
        return
    for fileinput_line in fileinput.input(sys.argv[1]):
        if 'Exit' == fileinput_line.rstrip():
            break
        line = fileinput_line.strip('\n')
        split_line = line.split('\t')
        holes = string_diff(split_line[0], split_line[1])
        single_hole_tokens = []
        single_hole_indices = []
        hole_tokens = holes[0].split(',')
        hole_indices = holes[1].split(',')
        # This part gets isolated holes
        if len(hole_indices) == 1:
            single_hole_tokens.append(hole_tokens[0])
            single_hole_indices.append(hole_indices[0])
        else:
            if int(hole_indices[0]) != int(hole_indices[1]) - 1:
                single_hole_tokens.append(hole_tokens[0])
                single_hole_indices.append(hole_indices[0])
            for i in range(1, len(hole_indices)-1):
                if (int(hole_indices[i]) != int(hole_indices[i-1]) + 1 and
                    int(hole_indices[i]) != int(hole_indices[i+1]) -1):
                    single_hole_tokens.append(hole_tokens[i])
                    single_hole_indices.append(hole_indices[i])
            if int(hole_indices[-1]) != int(hole_indices[-2]) + 1:
                single_hole_tokens.append(hole_tokens[-1])
                single_hole_indices.append(hole_indices[-1])
        holes = (single_hole_tokens, single_hole_indices)
        # Can remove it to get all holes
        sys.stdout.write(f"{line}\t{','.join(holes[0])}\t{','.join(holes[1])}\n")

if __name__ == '__main__':
    main()
