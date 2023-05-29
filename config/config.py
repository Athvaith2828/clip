import os

working_directory = r''

data_path = os.path.join(working_directory, 'data')

image_path = os.path.join(working_directory, data_path, 'image')

interface_path = os.path.join(working_directory, 'interface')

os.makedirs(interface_path,exist_ok=True)

log_path = os.path.join(working_directory, 'logs')

os.makedirs(log_path, exist_ok=True)

caption_path = os.path.join(working_directory, data_path, 'caption_single.csv')

train_set = os.path.join(data_path, 'train_set.csv')

val_set = os.path.join(data_path, 'val_set.csv')

test_set = os.path.join(data_path, 'test_set.csv')

image_size = 224

text_tokenizer = 'distilbert-base-uncased'

max_length = 200

train_debug = False

database_path = image_path

upload_path = os.path.join(interface_path, 'uploads')

result = os.path.join(interface_path, 'templates/results.html')