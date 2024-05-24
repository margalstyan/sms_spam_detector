import pandas as pd
from sklearn.metrics import f1_score
from src.model.data_processing import prepare_data


class NaiveBayes:
    """
    A Naive Bayes classifier for binary classification of spam and ham.

    Attributes:
    p_spam (float): Probability of a message being spam.
    p_ham (float): Probability of a message being ham.
    spams (dict): Dictionary of word probabilities in spam messages.
    hams (dict): Dictionary of word probabilities in ham messages.
    """

    def __init__(self):
        """
        Initializes the NaiveBayes classifier with default probabilities and empty dictionaries for spam and ham words.
        """
        self.p_spam = 0
        self.p_ham = 0
        self.spams = {}
        self.hams = {}

    def fit(self, X_train, y_train):
        """
        Trains the NaiveBayes classifier on the given training data.

        Parameters:
        X_train (list of str): List of training messages.
        y_train (Series): Corresponding labels for the training messages.
        """
        x = prepare_data(X_train)
        spams = {}
        hams = {}
        for row, t in zip(x, y_train):
            for word in row:
                if t == "spam":
                    spams[word] = spams.get(word, 1) + 1
                else:
                    hams[word] = hams.get(word, 1) + 1

        hams.update({k: 1 for k in spams.keys() if k not in hams})
        spams.update({k: 1 for k in hams.keys() if k not in spams})

        spam_total_words_count = sum(spams.values())
        ham_total_words_count = sum(hams.values())

        spams = {k: v / spam_total_words_count for k, v in spams.items()}
        hams = {k: v / ham_total_words_count for k, v in hams.items()}

        self.p_spam = y_train.value_counts()['spam'] / y_train.shape[0]
        self.p_ham = 1 - self.p_spam

        self.spams = spams
        self.hams = hams

    def predict(self, X_test):
        """
        Predicts the labels for the given test data.

        Parameters:
        X_test (list of str): List of test messages.

        Returns:
        list of int: Predicted labels for the test messages.
        """
        x = prepare_data(X_test)
        y = []
        for row in x:
            is_spam = self.p_spam
            is_ham = self.p_ham
            for word in row:
                if word in self.spams:
                    is_spam *= self.spams[word]
                    is_ham *= self.hams[word]
            y.append(1 if is_spam > is_ham else 0)
        return y

    def score(self, X_test, y_test):
        """
        Computes the F1 score of the classifier on the given test data.

        Parameters:
        X_test (list of str): List of test messages.
        y_test (Series): True labels for the test messages.

        Returns:
        float: The F1 score of the classifier.
        """
        if isinstance(y_test, list | tuple):
            y_test = pd.array(y_test)
        preds = self.predict(X_test)
        y = y_test
        y = y.map(lambda x: 0 if x == "ham" else 1)
        return f1_score(preds, y)
