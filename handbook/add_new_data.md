
# How to Add the Amazon Dataset to Legommenders

This guide show you how to add the Amazon Video Games dataset to the **Legommenders** framework.

**Note**: Any shell commands below should be run from the repository **root folder** (not handbook folder).

---

## Step 1: Download the Dataset

You can download the Amazon dataset from [this link](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/).  
In this tutorial, we will use the **5-core Video Games** dataset along with its metadata.


```shell
wget https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFilesSmall/Video_Games_5.json.gz
gzip -d Video_Games_5.json.gz

wget https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/metaFiles2/meta_Video_Games.json.gz
gzip -d meta_Video_Games.json.gz  

# Create folder and put data in folder
mkdir -p data/game_raw/
mv Video_Games_5.json data/game_raw/
mv meta_Video_Games.json data/game_raw/
```

## Step 2: Register the Dataset

Register the data folder by adding an entry to the .data configuration file:

```shell
echo -e "\namz = data/game_raw" >> .data
```

This set `data_dir="data/game_raw"` for `__init__` function in AmzProcessor.

## Step 3: Implement the Dataset Processor
You’ll now need to implement the AmzProcessor class.

A sample implementation is provided in `handbook/samples/amz_processor.py`.
You will need to put this file into the correct location (`processor/amz_processor.py`).

```shell
ln -s handbook/samples/amz_processor.py processor/amz_processor.py
```

## Step 4: Preprocess the Data
Next, we run preprocessing script:
```shell
python process.py --data amz
```

Download `glove` embedding.
```shell
python embed.py --model glove
```

## Step 5: Train model
Train the NAML model on pre-processed data
```shell
python trainer.py \
  --data handbook/samples/amz_config.yaml \
  --model config/model/naml.yaml \
  --hidden_size 256 \
  --lr 0.001 \
  --batch_size 64 \
  --item_page_size 0 \
  --embed config/embed/glove.yaml
```


# Data structure details
In this part, we provide note on the detail of pre-processed data folder structure.

There are three main table types: users (`users.parquet`), items (`items.parquet`), and interactions (`train.parquet`, `val.parquet` and `test.parquet`).

- Interaction tables: Each row consists of one interaction record, which include impression ID, user ID, item ID, and click or not label.
- User table: Each row consists of one user information, which includes his interaction history and negative item interactions list.
