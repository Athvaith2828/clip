import os

working_directory = r'C:\Users\shobi\Desktop\inkers\clip'

data_path = os.path.join(working_directory, 'data')

image_path = os.path.join(working_directory, data_path, 'image')

caption_path = os.path.join(working_directory, data_path, 'captions.csv')

train_set = os.path.join(data_path, 'train_set.csv')

val_set = os.path.join(data_path, 'val_set.csv')

test_set = os.path.join(data_path, 'test_set.csv')

train_debug = False