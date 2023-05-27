import pandas as pd
import numpy as np


from config import config as config

np.random.seed(42)

def data_split(dataframe):
    '''
    This Function helps to split data into train,val,test.
    :return:
    '''

    length = len(dataframe)
    test_length = int(length * 10 / 100)
    test_df = dataframe.iloc[length-test_length:, :]
    df = dataframe.iloc[:length-test_length, :]
    length = len(df)
    ids = np.arange(0,length)
    val_id = np.random.choice(ids, size=test_length,
                              replace=False)
    train_id = [id for id in ids if id not in val_id]
    train_df = df.loc[train_id].reset_index(drop=True)
    val_df = df.loc[val_id].reset_index(drop=True)
    return train_df, val_df, test_df

if __name__ == "__main__":

    dataframe = pd.read_csv(config.caption_path)

    if config.train_debug:
        dataframe = dataframe.iloc[:100, :]

    train_df, val_df, test_df = data_split(dataframe)

    train_df.to_csv(config.train_set)

    val_df.to_csv(config.val_set)

    test_df.to_csv(config.test_set)



