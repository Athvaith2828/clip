import pandas as pd
import numpy as np
import albumentations as A
import torch.utils.data
from transformers import DistilBertTokenizer
from loguru import logger

from config import config as config
from config import train_config as tcg
from data_loader.clipdataset import ImageTextDataset

np.random.seed(42)

logger.add(f'{config.log_path}/data_loader.log')

def data_split(dataframe):
    '''
    This Function helps to split data into train,val,test.
    :return:
    '''

    length = len(dataframe)
    logger.info(f"Total datapoints  {length}")
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

    logger.info(f"datapoints in {len(train_df)}")
    logger.info(f"datapoints in {len(val_df)}")
    logger.info(f"datapoints in {len(test_df)}")

    return train_df, val_df, test_df

def image_transform():
    '''
    To resize the image and noramalize pixels
    :return:
    '''
    return A.Compose(
        [
            A.Resize(config.image_size, config.image_size, always_apply=True),
            A.Normalize(max_pixel_value=255.0, always_apply=True),
        ]
    )

def loader(dataframe, mode='Train'):
    '''
     This Function helps to build train,val,df data_loader then we can pass to model in batches.
    :param dataframe: df contains images filenames ana captions
    :param mode: Used to differient train/ test & val
    :return: Dataloader ready to go.
    '''

    logger.info(f"Loading {mode} data")

    transform = image_transform()

    tokenizer = DistilBertTokenizer.from_pretrained(config.text_tokenizer)

    dataset = ImageTextDataset(dataframe['image'].values,
                               dataframe['caption'].values,
                               tokenizer=tokenizer,
                               transform=transform)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=tcg.batch_size,
        num_workers=tcg.n_worker,
        shuffle=True if mode =='Train' else False
    )

    logger.info(f" Loaded {mode} data")

    return dataloader

def main():

    logger.info('Reading captions csv....')

    dataframe = pd.read_csv(config.caption_path)

    if config.train_debug:
        dataframe = dataframe.iloc[:100, :]

    logger.info("Splittings data into Train,validation and test sets")

    train_df, val_df, test_df = data_split(dataframe)

    logger.info('Stroing train,validation and test sets')

    train_df.to_csv(config.train_set)

    val_df.to_csv(config.val_set)

    test_df.to_csv(config.test_set)

    logger.info("Completed")

if __name__ == "__main__":

    main()



