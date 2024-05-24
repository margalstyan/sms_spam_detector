import pytest
from src.model.model import NaiveBayes
from src.model.data_processing import extract_words
from src.model.utils import split_data
import pandas as pd


@pytest.fixture(scope='module')
def setup_data():
    df = pd.DataFrame({
        'v1': ['ham', 'spam', 'ham', 'spam'],
        'v2': [
            'Hi there, how are you?',
            'Win $1000 now, click here!',
            'Are we still on for tonight?',
            'Exclusive deal just for you, act now!'
        ]
    })
    X_train, X_test, y_train, y_test = split_data(df)
    nb = NaiveBayes()
    nb.fit(X_train, y_train)
    return nb, X_test, y_test

def test_prediction(setup_data):
    nb, X_test, _ = setup_data
    prediction = nb.predict(X_test)
    assert isinstance(prediction, list)

def test_score(setup_data):
    nb, X_test, y_test = setup_data
    score = nb.score(X_test, y_test)
    assert isinstance(score, float)

def test_extract_words():
    result = extract_words("Hi, this is a test! 123")
    expected = {'hi': 1, 'this': 1, 'is': 1, 'a': 1, 'test': 1, '!': 1, '123': 1}
    assert result == expected
