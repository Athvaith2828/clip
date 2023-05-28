import os

from config import config as cg

batch_size = 32

head_lr = 1e-3

image_encoder_lr = 1e-4

text_encoder_lr = 1e-5

weight_decay = 1e-3

patience = 1

factor = 0.8

epochs = 2

early_epoch = 2

early_stop_patience = 2

n_worker = int(os.cpu_count())

model_path = os.path.join(cg.data_path, 'model')

isExist = os.path.exists(model_path)

if not isExist:
    os.makedirs(model_path)

model_path = os.path.join(model_path, 'best.pt')

mapping_path = os.path.join(cg.data_path, 'model/mapping.pth')

plot_fig = os.path.join(cg.data_path, 'model/loss.png')