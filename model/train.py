import pandas as pd
import torch
import itertools
from tqdm import tqdm
from loguru import logger
import matplotlib.pyplot as plt

from config import config as cg
from config import train_config as tcg
from model.model import clip

from data_loader.main import loader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logger.add(f'{cg.log_path}/train.log')


class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]
def train(model,  train_loader, optimizer, lr_scheduler, step):
    '''

    :param model: clip model
    :param train_loader: train data
    :param optimizer: adam
    :param lr_scheduler: learning rate
    :param step: for lr_scheduler
    :return:
    '''
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for batch in tqdm_object:
        batch = {k: v.to(device) for k,v in batch.items() if k != 'caption'}
        optimizer.zero_grad()
        loss =model(batch)
        loss.backward()
        optimizer.step()
        if step == 'batch':
            lr_scheduler.step()
        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
    return loss_meter

def validation(model, val_loader):
    '''
     This Function is used to pass validation set to model to get val loss
    :param model: clip model
    :param val_loader: val_data
    :return: val_loss
    '''
    loss_meter = AvgMeter()

    tqdm_object = tqdm(val_loader, total=len(val_loader))
    for batch in tqdm_object:
        batch = {k:v.to(device) for k,v in batch.items() if k != 'caption'}
        loss = model(batch)
        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
    return loss_meter

def main():
    '''
    This function is used to train the clip model
    :return:
    '''
    logger.info("Starting training....")

    logger.info(f"Using {device}")

    train_df = pd.read_csv(cg.train_set)

    logger.info(f'Total Training DataPoints {len(train_df)}')

    val_df = pd.read_csv(cg.val_set)

    logger.info(f'Total validation DataPoints {len(val_df)}')

    train_loader = loader(train_df, 'Train')

    val_loader = loader(val_df, 'val')

    model = clip().to(device)
    params = [
        {"params": model.image_encoder.parameters(), "lr": tcg.image_encoder_lr},
        {"params": model.text_encoder.parameters(), "lr": tcg.text_encoder_lr},
        {"params": itertools.chain(
            model.image_projection.parameters(), model.text_projection.parameters()
        ), "lr": tcg.head_lr, "weight_decay": tcg.weight_decay}
    ]
    optimizer = torch.optim.AdamW(params, weight_decay=0.)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=tcg.patience, factor=tcg.factor
    )
    step = "epoch"

    total_train_loss , total_val_loss = [], []

    early_stop_counter = 0

    best_loss = float('inf')
    for epoch in range(tcg.epochs):
        logger.info(f"Epoch: {epoch + 1}")
        model.train()
        train_loss = train(model, train_loader, optimizer, lr_scheduler, step)
        total_train_loss.append(train_loss.avg)
        model.eval()
        with torch.no_grad():
            valid_loss = validation(model, val_loader)
        total_val_loss.append(valid_loss.avg)
        if valid_loss.avg < best_loss:
            early_stop_counter = 0
            best_loss = valid_loss.avg
            torch.save(model.state_dict(), tcg.model_path)
            logger.info("Saved Best Model!")
        else:
            early_stop_counter += 1

        lr_scheduler.step(valid_loss.avg)
        logger.info(f"Epoch: {epoch + 1}, training_loss: {train_loss.avg}, validation_loss: {valid_loss.avg}")

        # Early stopping condition
        if (epoch > tcg.early_epoch and
                early_stop_counter >=  tcg.early_stop_patience):
            logger.info(f"val loss is not improved in {tcg.early_stop_patience} epochs"
                        f"Early stopping ......")
            break

    x = range(1,len(total_train_loss)+1)
    plt.plot(x, total_train_loss, label='avg_train_loss')
    plt.plot(x, total_val_loss, label='avg_val_loss')

    plt.xlabel('Epochs')
    plt.ylabel('Avg_Loss')
    plt.title('Epochs Vs Avg loss')

    plt.legend()
    plt.savefig(tcg.plot_fig)

if __name__ == "__main__":
    main()


