from sklearn.model_selection import train_test_split

def split_data(df):
    """
    Splits the DataFrame into training and testing sets.

    Parameters:
    df (DataFrame): The DataFrame to split. It should contain features in one column and labels in another.

    Returns:
    tuple: Four elements - training features (X_train), testing features (X_test), training labels (y_train), testing labels (y_test).
    """
    # Split the data into training and testing sets
    # df['v2'] contains the features (text messages)
    # df['v1'] contains the labels (spam or ham)
    # test_size=0.2 means 20% of the data will be used for testing, and 80% for training
    X_train, X_test, y_train, y_test = train_test_split(df['v2'], df['v1'], test_size=0.2)
    return X_train, X_test, y_train, y_test
