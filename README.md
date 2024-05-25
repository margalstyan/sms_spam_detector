# Spam Classifier

## Overview
This project implements a spam classifier using a Naive Bayes model. The classifier is trained on the SMS Spam Collection Dataset.

## Project Structure
- `src/`: Contains the source code for model, server and tests.
- `model/`: Contains the data processing and the Naive Bayes model.
- `tests/`: Contains tests for the project.
- `main.py`: Script to run server.
- `requirements.txt`: List of dependencies.
- `README.md`: Project documentation.

## Setup
1. Clone the repository.
2. Install the dependencies: 
   ```bash
   pip install -r requirements.txt
3. Run the main script:
    ```bash
   python main.py

## Running Tests
To run all tests, use the following command:
```bash
pytest src/tests/
```

To run the tests for model:
```bash
pytest src/tests/ -m model
```

To run the tests for server:
```bash
pytest src/tests/ -m server
```


To run the async tests for server:
```bash
pytest src/tests/ -m asyncio
```




