from torch import nn
import torch.nn.functional as F
import timm
from transformers import DistilBertModel

from config import model_config as mcg

class ImageEncoder(nn.Module):
    '''
    Encodes images(Image embeddings) using pretrained model
    '''
    def __init__(self,
                 model_name=mcg.image_model,
                 pretrained=mcg.pretrained,
                 trainable=mcg.trainable):

        super.__init__()

        self.model = timm.create_model(
            model_name, pretrained
        )

        for param in self.model.parameters():
            param.requires_grad = trainable

    def forward(self, x):
        return self.model(x)

class TextEncoder(nn.Module):
    '''
    To get the text embedding for pretrained model
    '''

    def __init__(self, model_name=mcg.text_model,
                 pretrained=mcg.pretrained,
                 trainable=mcg.trainable):

        super.__init__()

        self.model = DistilBertModel.from_pretrained(
            model_name
        )

        for params in self.model.parameters():
            params.requires_grad = trainable

        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):

        out = self.model(input_ids=input_ids,
                         attention_mask=attention_mask)
        last_hidden = out.last_hidden_state

        return last_hidden[:, self.target_token_idx, :]

class projection(nn.Module):
    '''
     Used to fine tune context by reducing dim
    '''

    def __init__(self,
                 embedding_dim,
                 projection_dim=mcg.projection_dim,
                 dropout=mcg.dropout):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(x)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

