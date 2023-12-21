# Program to make a dataset from parallel translation data
# @author: Richard Yue

from bs4 import BeautifulSoup
import sys
import csv

def main():
    with open(sys.argv[1], 'rb') as fd:
        content = fd.read()
    soup = BeautifulSoup(content, 'xml')
    prop = soup.find('prop').text
    segments = soup.find_all('tu')
    en_fr_segments = [segment for segment in segments if 'EN-GB' in str(segment) and 'FR-FR' in str(segment)]
    en_segments = []
    fr_segments = []
    sys.stdout.write('en_text\tfr_text\n')
    for segment in en_fr_segments:
        sys.stdout.write(segment.find('tuv', {"xml:lang": 'EN-GB'}).text.strip('\n'))
        sys.stdout.write('\t')
        sys.stdout.write(segment.find('tuv', {"xml:lang": 'FR-FR'}).text.strip('\n'))
        sys.stdout.write('\n')
    
if __name__ == "__main__":
    main()
