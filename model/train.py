import pandas as pd

from config import config as cg
from config import train_config as tcg

from data_loader.main import loader

if __name__ == "__main__":

    train_df = pd.read_csv(cg.train_set)

    val_df = pd.read_csv(cg.val_set)

    train_loader = loader(train_df, 'Train')

    val_loader = loader(val_df, 'val')


