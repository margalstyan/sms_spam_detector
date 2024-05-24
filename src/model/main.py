from src.model.data_processing import load_data
from src.model.model import NaiveBayes
from src.model.utils import split_data
import os

# Load training data from CSV file
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
df = load_data(f'{ROOT_DIR}/train_data/spam.csv')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = split_data(df)

# Initialize the NaiveBayes model
nb = NaiveBayes()

# Train the model using the training data
nb.fit(X_train, y_train)

if __name__ == "__main__":
    # Evaluate the model on the test data
    score = nb.score(X_test, y_test)
    print(f"The model is ready to use with a test F1 score of: {score}")
