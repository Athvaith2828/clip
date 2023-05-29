import torch
import pandas as pd
from loguru import logger

from config import config as cg
from config import train_config as tcg
from data_loader.main import loader
from model.train import validation
from model.model import clip

device = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logger.add(f'{cg.log_path}/test.log')

def main():
    '''
    This Function is used to test model performance using test data
    :return:
    '''
    model = clip().to(device)

    model.load_state_dict(torch.load(tcg.model_path, map_location=device))

    test_df = pd.read_csv(cg.test_set)

    logger.info(f'Got {len(test_df)} to test')

    test_loader = loader(test_df, 'test')

    loss = validation(model, test_loader)

    logger.info(f"Avg Test loss is {loss.avg}")

if __name__ == "__main__":
    main()