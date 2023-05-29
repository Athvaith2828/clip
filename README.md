# clip

## This Module is used to train clip model. 

### Modules
    1. data_loader - It is used to prepare train,val,test split
    2. model - It is used to define,train and create embeddings
    3. interface - It is used to run the app and it predictions

### Setup
    1. Create a env
    2. pip install -r requirements.txt
    3. Change the data path in config(data and model is stored in data folder)
        . set the working_directory in config.py (C:/.../clip)

    4. Run [main.py](data_loader/main.py) to generate train,validation,test set.
    5. Run [train.py](model/train.py) to train the model
    6. Run [embeddings.py](model/embeddings.py) to get the embeddings from trained model
    7. Run [app.py](interface/app.py) for getting predictions via API
    8. Run [test.py](interface/test.py) to check the performance of the model.
    

