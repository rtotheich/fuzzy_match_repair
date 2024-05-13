# Program to split fuzzy matches
# into separate files for a given
# split
# @author: Richard Yue

import fileinput
import sys

def main():
    min = int(sys.argv[2])
    max = int(sys.argv[3])
    sys.stdout.write('segment\tbest\tscore\n')
    for line in fileinput.input(sys.argv[1]):
        if 'Exit' == line.rstrip():
            break
        score = line.split('\t')[2].strip('\n')
        try:
            if int(score) >= min and int(score) <= max:
                sys.stdout.write(line)
        except:
            continue

if __name__ == '__main__':
    main()
