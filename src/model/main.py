from src.model.data_processing import load_data
from src.model.model import NaiveBayes
from src.model.utils import split_data
import os

# Load train_data
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
df = load_data(f'{ROOT_DIR}/train_data/spam.csv')

# Split train_data
X_train, X_test, y_train, y_test = split_data(df)

# Train model
nb = NaiveBayes()
nb.fit(X_train, y_train)

if __name__ == "__main__":
    # Evaluate model
    score = nb.score(X_test, y_test)
    print(f"F1 Score: {score}")
