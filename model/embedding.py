import pandas as pd
import torch
from tqdm import tqdm
from loguru import logger

from config import config as cg
from config import train_config as tcg
from model import clip
from data_loader.main import loader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logger.add(f'{cg.log_path}/embedding.log')

def main():

    logger.info("Reading training data...")
    df = pd.read_csv(cg.train_set)

    logger.info(f"Total datapoints {len(df)}")

    logger.info("Loading the model....")
    model = clip().to(device)

    model.load_state_dict(torch.load(tcg.model_path, map_location=device))

    train_loader = loader(df, 'save')

    image_file = list(df['image'].values)

    captions = list(df['caption'].values)

    total_image_embeddings = []

    total_text_embeddings = []

    logger.info("Getting Embedddings for training data")

    model.eval()

    with torch.no_grad():
        for batch in tqdm(train_loader):

            image_feature = model.image_encoder(batch['image'].to(device))
            image_embeddings = model.image_projection(image_feature)
            total_image_embeddings.append(image_embeddings)

            text_feature = model.text_encoder(batch['input_ids'],
                                              batch['attention_mask'])
            text_embeddings = model.text_projection(text_feature)
            total_text_embeddings.append(text_embeddings)

    total_image_embeddings = torch.cat(total_image_embeddings)

    total_text_embeddings = torch.cat(total_text_embeddings)

    logger.info("Storing the mapping dict....")

    mappings = {}

    mappings['image_filename'] = image_file

    mappings['caption'] = captions

    mappings['image_embeddings'] = total_image_embeddings

    mappings['text_embeddings'] = total_text_embeddings

    torch.save(mappings, tcg.mapping_path)

    logger.info("Done")

if __name__ == "__main__":
    main()










