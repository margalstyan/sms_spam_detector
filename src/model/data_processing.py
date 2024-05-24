import pandas as pd
import re


def load_data(filepath):
    df = pd.read_csv(filepath, encoding='cp1252')
    df = df[['v1', 'v2']]
    return df


def extract_words(input_string):
    characters_dict = {}

    input_string = input_string.replace("'", "")
    word_pattern = r'\b[a-zA-Z]+\b'
    symbol_pattern = r'[$€£%^&/:,.*!@#]+'
    number_pattern = r'\b\d+\b'

    words = re.findall(word_pattern, input_string.lower())
    symbols = re.findall(symbol_pattern, input_string)
    numbers = re.findall(number_pattern, input_string)

    for word in words:
        characters_dict[word] = characters_dict.get(word, 0) + 1
    for symbol in symbols:
        characters_dict[symbol] = characters_dict.get(symbol, 0) + 1
    for number in numbers:
        characters_dict[number] = characters_dict.get(number, 0) + 1

    return characters_dict


def prepare_data(x):
    new_x = []
    for data in x:
        new_x.append(extract_words(data))
    return new_x
