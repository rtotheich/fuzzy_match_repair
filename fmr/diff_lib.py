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
    holes = []
    tokens = []
    differences = []
    for i,s in enumerate(difflib.ndiff(s2, s1)):
        if s[0]!='-':
            holes.append(s)

    for i,diff in enumerate(holes):
        if diff[0] == '+':
            tokens.append(diff[2:])
            differences.append(i)
    return (tokens, differences)

def main():
    s1 = "The man went to the store to buy groceries"
    s2 = "The man went to the deli to get a lot of groceries"
    print(f"Example new segment: {s1}")
    print(f"Example tm segment: {s2}")
    token_diff = string_diff(s1, s2)
    print(f"Tokens: {token_diff[0]}\nDifferences: {token_diff[1]}")

if __name__ == "__main__":
    main()
