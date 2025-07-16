# <img src="assets/lego.png" alt="icon" style="vertical-align: middle; height: 32px;"> Legommenders

> [**What is Legommenders?**](handbook/what-is-legommenders.md)  
> Legommenders is a content-based recommendation library designed for the era of large language models.  
> Click the title above to learn more.

## Handbooks

- [Training on xMIND dataset](handbook/training-on-xmind-dataset.md)
- [How to customize a new dataset processor](handbook/how-to-customize-a-new-dataset-processor.md) by [@chenxing1999](https://github.com/chenxing1999)

## ⚙️ Installation & Getting Started

1. **Clone the Repo:**

    ```shell
    gh repo clone Jyonn/Legommenders
    cd Legommenders
    ```

2. **Install Dependencies:**

    ```shell
    pip install -r requirements.txt
    ```
    
    Ensure you have Python 3.10+ and a properly set up PyTorch environment: Nvidia GPU, Apple MPS, or even CPU device (`--cuda -1`).  

3. **Prepare the Configurations** _(Optional)_

    To run the data preprocessing scripts and generate the datasets by yourself, you should download the raw dataset and add the path to the `.data` file:
    
    ```shell
    touch .data
    echo -e "\n <name> = /path/to/data" >> .data  # e.g., echo -e "\n mind = /path/to/mind" >> .data
    ```
    
    Legommenders will link `<name>` to a `class <Name>Processor` defined at `processor/*.py`.

    You can also define HuggingFace language models at `.model` file:

   ```shell
   echo -e "\n <name> = huggingface/path" >> .model  # e.g., llama3.1 = meta-llama/Llama-3.1-8B-Instruct-evals
   ```

4. **Run the Project:** Use command-line tools to preprocess data, train models, and evaluate performance:
    
    ```shell
    # e.g., python process.py --data mind 
    python process.py --data <name>
   
    # e.g., python trainer.py --data config/data/mind.yaml --model config/model/naml.yaml --batch_size 64 --lr 0.001 --hidden_size 256 
    python trainer.py --data config/data/<name>.yaml --model config/model/<model>.yaml --batch_size <batch-size> --lr <learning-rate> --hidden_size <hidden-size> 
    ```

## 📊 Supported Datasets

Legommenders supports 15+ datasets across domains like news, books, movies, music, fashion, and e-commerce. The supported datasets can be categorized into three groups:

- **Native**: Legommenders provide native dataset processing scripts, i.e., `processor/mind_processor.py`.
- **Bridge**: Other repositories (e.g., [RecBench](https://github.com/Jyonn/RecBench)) process this dataset into their format, and Legommenders provides a bridge to convert it into our format, i.e., `processor/recbench_processor.py`. Using such datasets can make cross-repository models easy to evaluate.
- **Community**: Users can design processors to convert unsupported datasets into Legommenders format.

**\* Single dataset can be supported by multiple channels.**

| Dataset                                                                                         | Version    | Name         | Domain     | Support     | Comment                                                                                                                |
|-------------------------------------------------------------------------------------------------|------------|--------------|------------|-------------|------------------------------------------------------------------------------------------------------------------------|
| [**MIND**](https://msnews.github.io/)                                                           | small      | mind         | News       | ✅ Native    | [View Processor](https://github.com/Jyonn/Legommenders/blob/master/processor/mind_processor.py)                        |
| [**MIND**](https://msnews.github.io/)                                                           | large      | mindlarge    | News       | ❌           | ❌ Call for Contribution                                                                                                |
| [**MIND**](https://msnews.github.io/)                                                           | small      | mindrb       | News       | ✅ Bridge    | [View Processor](https://github.com/Jyonn/Legommenders/blob/master/processor/mind_recbench_processor.py)               |
| [**MIND**](https://msnews.github.io/)                                                           | small      | oncemind     | News       | ✅ Native    | Used for [ONCE](https://github.com/Jyonn/ONCE) paper. [View Processor](https://github.com/Jyonn/ONCE/tree/main/LegoV2) |
| [**xMIND**](https://github.com/andreeaiana/xMIND/)                                              | small      | xmind        | News       | ✅ Native    | [View Processor](https://github.com/Jyonn/Legommenders/blob/master/processor/xmind_processor.py)                       |
| [**PENS**](https://msnews.github.io/pens)                                                       | N/A        | pensrb       | News       | ✅ Bridge    | [View Processor](https://github.com/Jyonn/Legommenders/blob/master/processor/pens_recbench_processor.py)               |
| [**Adressa**](https://reclab.idi.ntnu.no/dataset/)                                              | 1week      | adressa      | News       | ❌           | ❌ Call for Contribution                                                                                                |
| [**Adressa**](https://reclab.idi.ntnu.no/dataset/)                                              | 10week     | adressalarge | News       | ❌           | In Norway language. ❌ Call for Contribution                                                                            |
| [**EB-NeRD**](https://recsys.eb.dk/index.html)                                                  | N/A        | ebnerdrb     | News       | ✅ Bridge    | [View Processor](https://github.com/Jyonn/Legommenders/blob/master/processor/ebnerd_recbench_processor.py)             |
| [**Goodreads**](https://mengtingwan.github.io/data/goodreads)                                   | N/A        | goodreadsrb  | Book       | ✅ Bridge    | [View Processor](https://github.com/Jyonn/Legommenders/blob/master/processor/goodreads_recbench_processor.py)          |
| [**MovieLens**](https://grouplens.org/datasets/movielens/)                                      | Unknown    | movielensrb  | Movie      | ✅ Bridge    | [View Processor](https://github.com/Jyonn/Legommenders/blob/master/processor/movielens_recbench_processor.py)          |
| [**MicroLens**](https://github.com/westlake-repl/MicroLens)                                     | N/A        | microlensrb  | Movie      | ✅ Bridge    | [View Processor](https://github.com/Jyonn/Legommenders/blob/master/processor/microlens_recbench_processor.py)          |
| [**Netflix Prize**](https://www.kaggle.com/competitions/netflix-prize/data)                     | N/A        | netflixrb    | Movie      | ✅ Bridge    | [View Processor](https://github.com/Jyonn/Legommenders/blob/master/processor/netflix_recbench_processor.py)            |
| [**LastFM**](http://millionsongdataset.com/lastfm)                                              | N/A        | lastfmrb     | Music      | ✅ Bridge    | [View Processor](https://github.com/Jyonn/Legommenders/blob/master/processor/lastfm_recbench_processor.py)             |
| [**HotelRec**](https://github.com/zhaofangyuan98/HotelRec)                                      | N/A        | hotelrecrb   | Hotel      | ✅ Bridge    | [View Processor](https://github.com/Jyonn/Legommenders/blob/master/processor/hotelrec_recbench_processor.py)           |
| [**Yelp**](https://www.yelp.com/dataset)                                                        | N/A        | yelprb       | Restaurant | ✅ Bridge    | [View Processor](https://github.com/Jyonn/Legommenders/blob/master/processor/yelp_recbench_processor.py)               |
| [**H&M**](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations)     | N/A        | hmrb         | Fashion    | ✅ Bridge    | [View Processor](https://github.com/Jyonn/Legommenders/blob/master/processor/hm_recbench_processor.py)                 |
| [**POG**](https://drive.google.com/drive/folders/1tHCG8x1fLF18ccuXMJsEB8mNIn5n4F2l?usp=sharing) | N/A        | pogrb        | Fashion    | ✅ Bridge    | [View Processor](https://github.com/Jyonn/Legommenders/blob/master/processor/pog_recbench_processor.py)                |
| [**Amazon**](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/)                             | books      | booksrb      | Book       | ✅ Bridge    | [View Processor](https://github.com/Jyonn/Legommenders/blob/master/processor/books_recbench_processor.py)              |
| [**Amazon**](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/)                             | automotive | automotiverb | Automotive | ✅ Bridge    | [View Processor](https://github.com/Jyonn/Legommenders/blob/master/processor/automotive_recbench_processor.py)         |
| [**Amazon**](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/)                             | cds        | cdsrb        | Music      | ✅ Bridge    | [View Processor](https://github.com/Jyonn/Legommenders/blob/master/processor/cds_recbench_processor.py)                |
| [**Amazon**](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/)                             | games      | games        | Game       | ✅ Community | Processor Coming Soon!                                                                                                 |

Datasets can be processed into Legommenders format using built-in scripts based on RecBench. You can directly download the data from [here](https://drive.google.com/drive/folders/1PP2PMqg4Fxe8Qb2haob8eJy7g8bgw6tC?usp=sharing).

## 🏗️ Model Architecture & Algorithms

Legommenders is built with **a modular, layered architecture**:

- **Multimodal Dataset Processor**: Converts raw data into three unified tables (item content, user history, interactions) using UniTok.
- **Content Operator**: Encodes item content using static (e.g., Glove) or deep models (e.g., CNN, BERT, GPT). Supports 15+ content modules.
- **Behavior Operator**: Encodes user behavior history using methods like attention, RNN, Transformer. 8+ options available.
- **Click Predictor**: Predicts user-item interactions via dot product, MLP, DeepFM, DCN, etc.

The following models can be realized by Legommenders:

| Model                                                              | Type    | Config                               | Item Op    | User Op            | Predictor |
|--------------------------------------------------------------------|---------|--------------------------------------|------------|--------------------|-----------|
| [**NAML** (2019)](https://arxiv.org/abs/1907.05576)                | Recall  | `config/model/naml.yaml`             | CNN        | Additive Attention | Dot       |
| [**NRMS** (2019)](https://aclanthology.org/D19-1671/)              | Recall  | `config/model/nrms.yaml`             | Attention  | Attention          | Dot       |
| [**LSTUR** (2019)](https://aclanthology.org/P19-1033/)             | Recall  | `config/model/lstur.yaml`            | CNN        | GRU                | Dot       |
| [**PLM-NR** (2021)](https://arxiv.org/abs/2104.07413)              | Recall  | `config/model/bert-naml.yaml`        | BERT       | Additive Attention | Dot       |
| [**Fastformer** (2023)](https://arxiv.org/abs/2108.09084)          | Recall  | `config/model/fastformer.yaml`       | Fastformer | Fastformer         | Dot       |
| [**MINER** (2022)](https://aclanthology.org/2022.findings-acl.29/) | Recall  | `config/model/bert-miner.yaml`       | BERT       | PolyAttention      | Dot       | 
| [**ONCE** (2024)](https://arxiv.org/abs/2305.06566)                | Recall  | `config/model/llama-naml.yaml`       | Llama1     | Additive Attention | Dot       |
| [**IISAN** (2024)](https://arxiv.org/abs/2404.02059)               | Recall  | `config/model/llama-iisan-naml.yaml` | Llama1     | Additive Attention | Dot       |
| [**PNN** (2016)](https://arxiv.org/pdf/1611.00144)                 | Ranking | `config/model/pnn_id.yaml`           | N/A        | Pooling            | PNN       |
| [**DeepFM** (2017)](https://arxiv.org/abs/1703.04247)              | Ranking | `config/model/deepfm_id.yaml`        | N/A        | Pooling            | DeepFM    |
| [**DCN** (2017)](https://arxiv.org/pdf/1708.05123)                 | Ranking | `config/model/dcn_id.yaml`           | N/A        | Pooling            | DCN       |
| [**DIN** (2017)](https://arxiv.org/abs/1706.06978)                 | Ranking | `config/model/din_id.yaml`           | N/A        | N/A                | DIN       |
| [**AutoInt** (2018)](https://arxiv.org/abs/1810.11921)             | Ranking | `config/model/autoint_id.yaml`       | N/A        | Pooling            | AutoInt   |
| [**DCNv2** (2020)](https://arxiv.org/abs/2008.13535)               | Ranking | `config/model/dcnv2_id.yaml`         | N/A        | Pooling            | DCNv2     |
| [**MaskNet** (2021)](https://arxiv.org/abs/2102.07619)             | Ranking | `config/model/masknet_id.yaml`       | N/A        | Pooling            | MaskNet   |
| [**GDCN** (2023)](https://arxiv.org/abs/2311.04635)                | Ranking | `config/model/gdcn_id.yaml`          | N/A        | Pooling            | GDCN      | 
| [**FinalMLP** (2023)](https://arxiv.org/abs/2304.00902)            | Ranking | `config/model/finalmlp_id.yaml`      | N/A        | Pooling            | FinalMLP  |


## 🚀 Training & Evaluation

1. **Data Preprocessing:**

```shell
python process.py --data mind
```

2. **Embedding Setup (e.g. BERT):**

```shell
python embed.py --model bertbase
```

3. **Train a Model:**

```shell
python trainer.py \ 
  --data config/data/mind.yaml \
  --model config/model/naml.yaml \
  --hidden_size 256 \ 
  --lr 0.001 \ 
  --batch_size 64 \
  --item_page_size 0 \
  --embed config/embed/glove.yaml
```

The default evaluation metric on the validation set is GAUC. You can specify other metrics like MRR, NDCG, by adding `--metrics mrr` or `--metrics ndcg@10` to the command. We will list all our supported metrics below.

4. **Evaluate:** 

After trained, Trainer will automatically evaluate the model on test dataset. You can also use the `tester.py` script to load saved models and evaluate them.

By default, the evaluation metrics on the test dataset includes:
  - GAUC
  - MRR
  - NDCG@1
  - NDCG@5
  - NDCG@10

We also support the following evaluation metrics:
  - LogLoss
  - AUC
  - LRAP
  - F1@threshold
  - HitRatio@k
  - Recall@k

You can add the evaluation metrics at `utils/metrics.py`.

⚠️ **NOTE**: following existing recommendation repositories, the implementation of `MRR` is not the same as the original one. To get the original MRR, use `MRR0` instead (HIGHLY RECOMMEND).

## 🧪 Example Command

### Train NAML model on MIND:

```shell
python trainer.py \ 
  --data config/recbench/mind.yaml \
  --model config/model/bert-naml.yaml \
  --hidden_size 256 \
  --lr 0.001 \
  --batch_size 64 \
  --lm glove \
  --embed config/embed/glove.yaml
```

### To use BERT instead of GloVe

```shell
python trainer.py \ 
  --data config/recbench/mind.yaml \
  --model config/model/bert-naml.yaml \
  --hidden_size 256 \
  --lr 0.0001 \
  --batch_size 64 \
  --lm glove \
  --embed config/embed/bert.yaml \  # generate the yaml first, by running python embed.py --model bertbase
  --item_page_size \  # set it as large as possible based on your GPU memory  
  --use_lora true \
  --lora_r 8 \
  --lora_alpha 128 \
  --tune_from -2  # freeze the first N-1 layers, and tune the last layer, it is the same as --tune_from 10
```

#### [ONCE-DIRE-LLAMA1-NAML](https://arxiv.org/abs/2305.06566)

```bash
python trainer.py 
  --data config/data/mind-lm-prompt.yaml \  # for more powerful language models, we suggest to use the data concatenated with natural prompts
  --model config/model/llama-naml.yaml \ 
  --hidden_size 256 \ 
  --lr 0.0001 \
  --batch_size 64 \
  --item_page_size 64 \
  --embed config/embed/llama.yaml \ # generate the yaml first, by running python embed.py --model llama1
  --use_lora 1 \
  --lora_r 32 \
  --lora_alpha 128 \ 
  --lm llama1 \
  --llama 1 \
  --tune_from -2  # freeze the first N-1 layers, and tune the last layer, it is the same as --tune_from 30
```

## Updates

- **2025-07-14**: Code comments are available!
- **2025-04-10**: New LLM Adaptor: IISAN is supported. 
- **2025-02-18**: Legommenders v2.0, with multiple LLMs support, simplified configuration, more CTR predictors, and recbench-based datasets is released!
- **2025-01-06**: Legommenders v2.0 beta is released!
- **2024-12-05**: LSTUR model is now re-added to the Legommenders package, which was not compatible from Jan. 2024.
- **2024-01-23**: Legommenders partially supports the flatten sequential recommendation model. New models are added, including: MaskNet, GDCN, etc.
- **2023-10-16**: We clean the code and convert names of the item-side parameters.
- **2023-10-05**: The first recommender system package with a modular-design, Legommenders, is released!
- **2022-10-22**: Legommenders project is initiated.

## Citations

If you find Legommenders useful in your research, please consider citing our project:

```
@misc{legommenders,
  title={Legommenders: A Comprehensive Content-Based Recommendation Library with LLM Support},
  author={Liu, Qijiong and Fan, Lu and Wu, Xiao-Ming},
  booktitle = {Proceedings of the ACM Web Conference 2025},
  month = {may},
  year = {2025},
  address = {Australia, Sydney},
}
```

Thank you for your interest in Legommenders! Feel free to raise issues or contribute 🙏. Happy Recommending!

## Acknowledgement

We would like to thank Jieming Zhu and [FuxiCTR](https://github.com/reczoo/FuxiCTR) project for providing multiple useful CTR predictors.

We would like to thank [transformers](https://huggingface.co/transformers/) for providing the pre-trained language models.

We would like to thank [UniTok V4](https://unitok.qijiong.work/) for providing the unified data tokenization service.

We would like to thank [RecBench](https://github.com/RecBench) for providing unified recommendation dataset preprocessing framework.

We would like to thank [Oba](https://pypi.org/project/oba/), [RefConfig](https://pypi.org/project/refconfig/), and [SmartDict](https://pypi.org/project/smartdict/) for providing useful tools for our project.