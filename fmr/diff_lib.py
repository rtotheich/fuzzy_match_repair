# Get difference between two strings
# @author: Richard Yue

import difflib

def remove_chars(s:str, chars:list):
    for char in chars:
        s = s.replace(char, '')
    return s

def string_diff(s1, s2):
    s1 = remove_chars(s1, [',', '.', '!', '"', "'", '“', '”', '’', '`', '(', ')', '?', '/'])
    s2 = remove_chars(s2, [',', '.', '!', '"', "'", '“', '”', '’', '`', '(', ')', '?', '/'])
    s1 = s1.split()
    s2 = s2.split()
    differences = []
    tokens = []
    for i,s in enumerate(difflib.ndiff(s2, s1)):
        if s[0]==' ': continue
        elif s[0]=='+':
            differences.append(i-1)
            tokens.append(s[2:])
    print(f'Tokens: {tokens}')
    print(f'Indices: {differences}')

def main():
    s1 = "The man went to the store to buy groceries"
    s2 = "The man went to the deli to get groceries"
    print(f"Example new segment: {s1}")
    print(f"Example tm segment: {s2}")
    string_diff(s1, s2)

if __name__ == "__main__":
    main()
