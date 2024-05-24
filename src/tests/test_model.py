import os
import pytest
from src.model.data_processing import extract_words, load_data, prepare_data
from src.model.main import nb


@pytest.fixture(scope='module')
def data():
    messages = [
        'Hi there, how are you?',
        'Win $1000 now, click here!',
        'Are we still on for tonight?',
        'Exclusive deal just for you, act now!'
    ]
    predictions = [0, 1, 0, 0]
    return messages, predictions


@pytest.fixture(scope='module')
def prepared_data():
    return [{',': 1, 'are': 1, 'hi': 1, 'how': 1, 'there': 1, 'you': 1},
            {'!': 1, '$': 1, ',': 1, '1000': 1, 'click': 1, 'here': 1, 'now': 1, 'win': 1},
            {'are': 1, 'for': 1, 'on': 1, 'still': 1, 'tonight': 1, 'we': 1},
            {'!': 1, ',': 1, 'act': 1, 'deal': 1, 'exclusive': 1, 'for': 1, 'just': 1, 'now': 1, 'you': 1}]


def test_prediction(data):
    messages, predictions = data
    model_predict = nb.predict(messages)
    assert isinstance(model_predict, list)
    assert model_predict == predictions


def test_score(data):
    messages, predictions = data
    score = nb.score(messages, predictions)
    assert isinstance(score, float)
    assert 0 <= score <= 1


def test_extract_words():
    # Simple test
    result = extract_words("Hi, this is a test! 123")
    expected = {'hi': 1, 'this': 1, 'is': 1, 'a': 1, 'test': 1, '!': 1, '123': 1, ',': 1}
    assert result == expected

    # test with multiple words
    result = extract_words("Hi there, how are you?, Hi, I am fine, and you?")
    expected = {'hi': 2, 'there': 1, 'how': 1, 'are': 1, 'you': 2, ',': 4, 'am': 1, 'i': 1, 'fine': 1, 'and': 1}
    assert result == expected


def test_load_data():
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    df = load_data(f'{ROOT_DIR}/../model/train_data/spam.csv')
    assert df.shape == (5572, 2)
    assert df['v1'].value_counts()['spam'] == 747
    assert df['v1'].value_counts()['ham'] == 4825
    assert 'v1' in df.columns and 'v2' in df.columns
    assert len(df.columns) == 2


def test_prepare_data(data, prepared_data):
    messages, predictions = data
    print(prepare_data(messages))
    assert prepared_data == prepare_data(messages)
