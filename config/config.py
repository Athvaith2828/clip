import os

working_directory = r'C:\Users\shobi\Desktop\inkers\clip'

data_path = os.path.join(working_directory, 'data')

image_path = os.path.join(working_directory, data_path, 'image')

log_path = os.path.join(working_directory, 'logs')

caption_path = os.path.join(working_directory, data_path, 'captions.csv')

train_set = os.path.join(data_path, 'train_set.csv')

val_set = os.path.join(data_path, 'val_set.csv')

test_set = os.path.join(data_path, 'test_set.csv')

image_size = 224

text_tokenizer = 'distilbert-base-uncased'

max_length = 200

train_debug = False

database_path = r'C:\Users\shobi\Desktop\inkers\clip\data\image'

upload_path = r'C:\Users\shobi\Desktop\inkers\clip\interface\uploads'