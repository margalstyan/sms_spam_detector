import pandas as pd
import numpy as np
import re

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

df = pd.read_csv('spam.csv', encoding='cp1252')
df = df[['v1', 'v2']]
X_train, X_test, y_train, y_test = train_test_split(df['v2'], df['v1'], test_size=0.2, random_state=42)

def extract_words(input_string):
    characters_dict = {}

    # Remove ' symbols not to separate one word further
    input_string = input_string.replace("'", "")

    # define regex patterns for words, symbols, and numbers
    word_pattern = r'\b[a-zA-Z]+\b'
    symbol_pattern = r'[$€£%^&/:,.*!@#]+'
    number_pattern = r'\b\d+\b'

    # find all words, symbols, and numbers in the input string
    words = re.findall(word_pattern, input_string.lower()) # saving all words in lowercase not to have duplications
    symbols = re.findall(symbol_pattern, input_string)
    numbers = re.findall(number_pattern, input_string)

    # Count occurrencies of each word, symbol and number
    for word in words:
        characters_dict[word] = characters_dict.get(word, 0) + 1
    for symbol in symbols:
        characters_dict[symbol] = characters_dict.get(symbol, 0) + 1
    for number in numbers:
        characters_dict[number] = characters_dict.get(number, 0) + 1

    # Return dictionary containing the extracted characters and their occurrencies
    return characters_dict

class NaiveBayes:
    def __init__(self):
        self.p_spam = 0
        self.p_ham = 0
        self.spams = {}
        self.hams = {}

    def prepare_data(self,x):
        # Return an array for the given X (train or test train_data) to store extracted words along with their counts.
        new_x = []
        for data in x:
            new_x.append(extract_words(data))
        return new_x

    def fit(self,X_train, y_train):
        x = self.prepare_data(X_train)
        # Will calculate probability of each word given spam/ham here (p(x|class))
        spams = {}
        hams = {}
        for row, t  in zip(x, y_train):
            for word in row:
                if t == "spam":
                    # Please note that initial value is set to be 1 not 0, see explanation in next comment
                    spams[word] = spams.get(word, 1) + 1
                else:
                    hams[word] = hams.get(word, 1) + 1

        # I am merging my 2 sets, not to have a word which exists in  e.g. hams,
        # and does not exist in spams, as it can result to case, when because a spam mail
        # contains a word, which does not exist in my spams dictonary, (p = 0),
        # then in the prediction step I'll have is_spam = 0, and the prediction will be wrong
        #
        # For that purpose, after merging 2 sets, I need to add 1 to each value, not to have 0s
        # And that's why I have set initial value 1 (pre-added that 1 to values)
        # And here while merging values, I am setting 1 values to them
        hams.update({k: 1 for k in spams.keys() if k not in hams})
        spams.update({k: 1 for k in hams.keys() if k not in spams})

        spam_total_words_count = sum(spams.values())
        ham_total_words_count = sum(hams.values())

        spams = {k: v / spam_total_words_count for k, v in spams.items()}
        hams = {k: v / ham_total_words_count for k, v in hams.items()}

        # calculate p(class)
        self.p_spam = y_train.value_counts()['spam']/y_train.shape[0]
        self.p_ham = 1 - self.p_spam

        self.spams = spams
        self.hams = hams

    def predict(self,X_test):
        x = self.prepare_data(X_test)
        y = []
        for row in x:
            is_spam = self.p_spam
            is_ham = self.p_ham
            for word in row:
                if word in self.spams:
                    # p(class) is proper to p(class)*Likelyhood (p(x_1|class)*...*p(x_n|class))
                    is_spam *= self.spams[word]
                    is_ham *= self.hams[word]
            # 1 is spam; 0 is ham
            y.append(1 if is_spam > is_ham else 0)
        return y

    def score(self,X_test,y_test):
        preds = self.predict(X_test)
        y = y_test
        y = y.map(lambda x: 0 if x == "ham" else 1)
        return f1_score(preds, y)


nb = NaiveBayes()
nb.fit(X_train, y_train)
nb.score(X_test, y_test)
nb.predict(["hi Mariam, how are you?"])
nb.predict(["Hi Mr. Agasi, I want to share with you a bussiness idea to earn up to $5000, to do it click on following link :  www.mylink.com"])