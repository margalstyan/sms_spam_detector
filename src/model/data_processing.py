import pandas as pd
import re

def load_data(filepath):
    """
    Loads data from a CSV file, extracts specific columns, and returns a DataFrame.

    Parameters:
    filepath (str): The path to the CSV file.

    Returns:
    DataFrame: A pandas DataFrame containing the specified columns.
    """
    df = pd.read_csv(filepath, encoding='cp1252')
    df = df[['v1', 'v2']]
    return df

def extract_words(input_string):
    """
    Extracts words, symbols, and numbers from an input string and counts their occurrences.

    Parameters:
    input_string (str): The string to extract words, symbols, and numbers from.

    Returns:
    dict: A dictionary with words, symbols, and numbers as keys and their counts as values.
    """
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
    """
    Prepares data by extracting words, symbols, and numbers from each string in the input list.

    Parameters:
    x (list of str): A list of strings to process.

    Returns:
    list of dict: A list of dictionaries, each containing words, symbols, and numbers as keys and their counts as values.
    """
    new_x = []
    for data in x:
        new_x.append(extract_words(data))
    return new_x
