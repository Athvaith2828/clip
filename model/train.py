import pandas as pd
import torch
import itertools
from tqdm import tqdm


from config import config as cg
from config import train_config as tcg
from model import clip

from data_loader.main import loader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    train_df = pd.read_csv(cg.train_set)

    val_df = pd.read_csv(cg.val_set)

    train_loader = loader(train_df, 'Train')

    val_loader = loader(val_df, 'val')

    print('data loaded')
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

    best_loss = float('inf')
    for epoch in range(tcg.epochs):
        print(f"Epoch: {epoch + 1}")
        model.train()
        train_loss = train(model, train_loader, optimizer, lr_scheduler, step)
        model.eval()
        with torch.no_grad():
            valid_loss = validation(model, val_loader)

        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            torch.save(model.state_dict(), "best.pt")
            print("Saved Best Model!")

        lr_scheduler.step(valid_loss.avg)


if __name__ == "__main__":
    main()


