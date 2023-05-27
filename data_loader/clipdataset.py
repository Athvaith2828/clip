import torch
import cv2

from config import config as config


class ImageTextDataset(torch.utils.data.Dataset):
    '''
    To create a dataset for clip model: each datapoint will have
        1. transformed image
        2. tokenized input ids
        3. attention mask
        4. caption.

    '''
    def __init__(self, file_name, caption, tokenizer, transform):

        self.file_name = file_name

        self.caption = list(caption)

        self.transform = transform

        self.tokenized_captions = tokenizer(
            self.caption, padding=True, truncation=True,
            max_length= config.max_length
        )

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(value[idx])
            for key, value in self.tokenized_captions.items()
        }

        image = cv2.imread(f"{config.image_path}/{self.file_name[idx]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image=image)['image']
        item['image'] = torch.tensor(image).permute(2, 0, 1).float()
        item['caption'] = self.caption[idx]

        return item

    def __len__(self):
        return len(self.caption)





