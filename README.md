# <img src="assets/lego.png" alt="icon" style="vertical-align: middle; height: 32px;"> Legommenders v2: A Modular Framework for Content-Based Recommendation in the Era of LLMs

**Legommenders** is an open-source library for content-based recommendation that supports “Lego-style” modular composition 🧱. It enables joint training of content encoders and user behavior models, integrating content understanding seamlessly into the recommendation pipeline. Researchers and developers can use Legommenders to easily assemble **thousands of different recommendation models** and run experiments across more than 15 datasets. Notably, Legommenders **pioneers integration with Large Language Models (LLMs)**, allowing LLMs to be used both as content encoders and as generators for data augmentation to build more personalized and effective recommenders 🎉.

## 🧠 Project Overview

The name "Legommenders" comes from "LEGO + Recommenders," symbolizing the idea of building recommendation models like Lego bricks 🫠. This project, proposed by The Hong Kong Polytechnic University, is the official implementation of the WWW 2025 paper, _Legommenders: A Comprehensive Content-Based Recommendation Library with LLM Support_. The key goal is to provide **a unified and flexible research framework** for content-driven recommendation. Traditional recommender systems often rely on static ID representations and struggle with cold-start problems. Legommenders focuses on content features (e.g., article texts, product descriptions) to enhance recommendations.

### Highlights:

- **Joint Modeling of Content & Behavior**: Supports end-to-end training of content encoders and user behavior models, ensuring content representations are task-aware.
- **Modular Design**: Provides LEGO-style composable modules for content processing, behavior modeling, prediction, etc.
- **Rich Built-in Models**: Includes classic models like NAML, NRMS, LSTUR, DeepFM, DCN, DIN, enabling rapid experimentation and comparison.
- **LLM Integration**: Enables LLMs (e.g., BERT, GPT, LLaMA) as content encoders or for data generation. Includes LoRA support for efficient fine-tuning.
- **Widely Adopted**: Already supports multiple research projects such as [ONCE](https://arxiv.org/abs/2305.06566), [SPAR](https://arxiv.org/abs/2402.10555),[GreenRec](https://arxiv.org/abs/2403.04736), and [UIST](https://arxiv.org/abs/2403.08206).

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

Ensure you have Python 3.x and a properly set up PyTorch environment (CUDA optional).

3. **Run the Project:** Use command-line tools to preprocess data, train models, and evaluate performance.

## 📊 Supported Datasets

Legommenders supports 15+ datasets across domains like news, books, movies, music, fashion, and e-commerce:

- 📰 MIND: Large-scale Microsoft news data for CTR prediction.
- 📰 PENS: Personalized news recommendation dataset.
- 📰 Adressa: News reading logs from Norway.
- 📰 EB-NeRD: RecSys Challenge 2024 news dataset.
- 📚 Goodreads: Book reviews and metadata.
- 📚 Amazon Books: Subset of Amazon product reviews.
- 🎥 MovieLens: Classic movie rating dataset.
- 📺 MicroLens: MovieLens dataset with user-item interactions.
- 📺 Netflix Prize: Large-scale movie rating competition dataset.
- 🎵 Amazon CDs: Music CD reviews and metadata.
- 🎵 Last.fm: Music playback logs and tagging data.
- 👗 H&M: Apparel and fashion product data.
- 👗 POG: Fashion product reviews and metadata.
- 📱 Amazon Electronics: Electronics product reviews and metadata.
- 🎮 Steam: Video game reviews and metadata.
- 🏨 HotelRec: Hotel recommendation dataset.
- ️️🍽️ Yelp: Restaurant reviews and metadata.

Datasets can be processed into Legommenders format using built-in scripts based on RecBench. You can directly download the data from [here](https://drive.google.com/drive/folders/1PP2PMqg4Fxe8Qb2haob8eJy7g8bgw6tC?usp=sharing).

## 🏗️ Model Architecture & Algorithms

Legommenders is built with **a modular, layered architecture**:

- **Multimodal Dataset Processor**: Converts raw data into three unified tables (item content, user history, interactions) using UniTok.
- **Content Operator**: Encodes item content using static (e.g., Glove) or deep models (e.g., CNN, BERT, GPT). Supports 15+ content modules.
- **Behavior Operator**: Encodes user behavior history using methods like attention, RNN, Transformer. 8+ options available.
- **Click Predictor**: Predicts user-item interactions via dot product, MLP, DeepFM, DCN, etc.

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

- **2025-04-10**: New LLM Adaptor: IISAN is supported. 
- **2025-02-18**: Legommenders v2.0, with multiple LLMs support, simplified configuration, more CTR predictors, and recbench-based datasets is released!
- **2025-01-06**: Legommenders v2.0 beta is released!
- **2024-12-05**: LSTUR model is now re-added to the Legommenders package, which was not compatible from Jan. 2024.
- **2024-01-23**: Legommenders partially supports the flatten sequential recommendation model. New models are added, including: MaskNet, GDCN, etc.
- **2023-10-16**: We clean the code and convert names of the item-side parameters.
- **2023-10-05**: The first recommender system package, Legommenders, with a modular-design is released!
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