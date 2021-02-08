import pandas as pd
import os

dir_name = os.path.dirname(os.path.abspath(__file__))


def load_data():
    train_df = pd.read_csv(dir_name + '/train.csv')
    train_x = train_df.drop('Survived', axis=1)
    train_y = train_df['Survived']
    test_x = pd.read_csv(dir_name + '/test.csv')
    return train_x, train_y, test_x
