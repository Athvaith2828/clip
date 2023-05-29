import torch
import cv2
from loguru import logger
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from tqdm import tqdm

from config import config as cg
from config import train_config as tcg
from model.model import clip
from data_loader.main import image_transform

logger.add(f'{cg.log_path}/pred.log')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logger.info(f"Running in {device}")
logger.info('Loading Model.....')

model = clip().to(device)
model.load_state_dict(torch.load(tcg.model_path, map_location=device))
logger.info('Model Loaded')

logger.info("Getting Mappings")

mappings = torch.load(f'{tcg.mapping_path}', map_location=device)

transform = image_transform()

def caption(image_embeddings):
    '''
    This Function helps to predict caption for the images
    :param images: Embeddings of the images
    :return: sutiable caption
    '''

    logger.info("Loading text embeddings...")

    text_embeddings = mappings['text_embeddings']

    text_embeddings = mappings['text_embeddings']
    captions = mappings['caption']

    image_embeddings_n = F.normalize(image_embeddings, p=2, dim=-1)
    text_embeddings_n = F.normalize(text_embeddings, p=2, dim=-1)

    dot_sim = image_embeddings_n @ text_embeddings_n.T

    value, index = torch.topk(dot_sim, 3)

    results = [captions[i] for i in index[:,1]]
    return results

def similar_images(target_embeddings):

    logger.info("Loading image_embeddings")

    image_embeddings = mappings['image_embeddings']

    image_names = mappings['image_filename']

    image_embeddings_n = F.normalize(image_embeddings, p=2, dim=-1)
    target_embeddings_n = F.normalize(target_embeddings, p=2, dim=-1)

    dot_sim = target_embeddings_n @ image_embeddings_n.T

    value, index = torch.topk(dot_sim, 6)

    results = []
    for ind in index:
        one_res = [f'{cg.database_path}\{image_names[i]}' for i in ind]
        results.append(one_res)
    return results

def main(file_names):

    logger.info(f"Predicting for {len(file_names)} images..")

    logger.info("Loading Images and Reforming.... ")

    images = []
    for file in file_names:
        image = cv2.imread(file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = transform(image=image)['image']
        image = torch.tensor(image).permute(2, 0, 1).float()
        images.append(image)

    images = torch.stack(images)

    batch_size = 32

    data_loader = DataLoader(TensorDataset(images), batch_size=batch_size)

    image_embeddings = []

    for batch in tqdm(data_loader):
        image_feature = model.image_encoder(batch[0])
        image_feature = model.image_projection(image_feature)
        image_embeddings.append(image_feature)
    image_embeddings = torch.cat(image_embeddings)
    caps = caption(image_embeddings)
    logger.info(caps)
    results = similar_images(image_embeddings)
    logger.info(results)
    return caps, results

if __name__ == "__main__":
    main()
