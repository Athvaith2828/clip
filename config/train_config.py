import os

import config as cg

batch_size = 512

n_worker = int(os.cpu_count())

model_path = os.path.join(cg.data_path, 'model')

isExist = os.path.exists(model_path)

if not isExist:
    os.makedirs(model_path)

model_path = os.path.join(model_path,'best.pt')