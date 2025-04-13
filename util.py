import pandas as pd
import sklearn

def train_val_test_split(df_input, df_target: pd.DataFrame, ratio: float, seed: int):
    """
    Performs the the sklearn train_test_split twice to sepearte a test and secondly a validation set 
    from the initial given datasets for input and target datasets with given ratio.
    :param dataframe: pandas DataFrame to process
    :param ratio: ratio of test and validation set
    :param seed: random state value (seed)
    """
    input_train_val, input_test, label_train_val, label_test = sklearn.model_selection.train_test_split(
        df_input, df_target, test_size=ratio, random_state=seed)

    # Split train_val set into train and val
    input_train, input_val, label_train, label_val = sklearn.model_selection.train_test_split(
        input_train_val, label_train_val, test_size=ratio, random_state=seed)
    
    return input_train, label_train, input_val, label_val, input_test, label_test