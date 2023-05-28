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

        super().__init__()

        self.model = timm.create_model(
            model_name, pretrained,
            num_classes=0, global_pool="avg"
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

        super().__init__()

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

    def __init__(
            self,
            embedding_dim,
            projection_dim=mcg.projection_dim,
            dropout=mcg.dropout
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

class clip(nn.Module):
    '''
     Matches image and text embeddings
    '''

    def __init__(self,
                 temperature = mcg.temperature,
                 image_embedding_dim = mcg.image_dimension,
                 text_embedding_dim = mcg.text_dimension):

        super().__init__()

        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()

        self.image_projection = projection(embedding_dim=image_embedding_dim)

        self.text_projection = projection(embedding_dim=text_embedding_dim)

        self.temperature = temperature

    def forward(self, batch):

        encoded_image = self.image_encoder(batch['image'])

        image_embeddings = self.image_projection(encoded_image)

        encoded_text = self.text_encoder(input_ids = batch['input_ids'],
                                       attention_mask = batch['attention_mask'])

        text_embeddings = self.text_projection(encoded_text)

        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        texts_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss = (images_loss + texts_loss) / 2.0  # shape: (batch_size)
        return loss.mean()

def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()