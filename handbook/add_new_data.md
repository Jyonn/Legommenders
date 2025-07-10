

This is the note on how to add Amazon dataset to Legommenders.

First, we need to download the data from: [Link](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/). I use the 5-core Videos games. Let's get both the 5-core and metadata then extract it.


Note: Any shell script is intended to be performed in repository root folder (not handbook folder).

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

Next, let's append this folder information to `.data`.
```shell
echo amz = data/game_raw >> .data
```

This set `data_dir="data/game_raw"`. Now let's implement `class AmzProcessor`. Sample file is provided in `handbook/samples/amz_processor.py`. You will need to put this file into `processor/amz_processor.py`.

```shell
ln -s handbook/samples/amz_processor.py processor/amz_processor.py
```

Next, we run preprocessing script:
```shell
python process.py --dataset amz
```

Download `glove` embedding.
```shell
python embed.py --model glove
```






# Data structure details
In this part, we provide note on the detail of pre-processed data folder structure.

There are three main table types: users (`users.parquet`), items (`items.parquet`), and interactions (`train.parquet`, `val.parquet` and `test.parquet`).

- Interaction tables: Each row consists of one interaction record, which include impression ID, user ID, item ID, and click or not label.
- User table: Each row consists of one user information, which includes his interaction history and negative item interactions list.
