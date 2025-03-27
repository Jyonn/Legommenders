# <img src="assets/lego.png" alt="icon" style="vertical-align: middle; height: 32px;"> Legommenders v2

*A Modular Framework for Recommender Systems in the Era of LLMs*

## Installation

```bash
gh repo clone Jyonn/Legommenders
cd Legommenders
pip install -r requirements.txt
```

## Quick Start with Three Steps

### Download Preprocessed Data

You can download the data from [here](https://drive.google.com/drive/folders/1PP2PMqg4Fxe8Qb2haob8eJy7g8bgw6tC?usp=sharing).

### Data Preprocessing (Optional)

```bash
python process.py --data mind
```

### Token Embedding Extraction

```bash
python embed.py --model bertbase
```

### Training a Recommender

#### [NAML](https://arxiv.org/abs/1907.05576)

```bash
python trainer.py --data config/data/mind.yaml --model config/model/naml.yaml --hidden_size 256 --lr 0.001 --batch_size 64 --item_page_size 0 --embed config/embed/glove.yaml
```

#### [NRMS](https://aclanthology.org/D19-1671/)

```bash
python trainer.py --data config/data/mind.yaml --model config/model/nrms.yaml --hidden_size 256 --lr 0.001 --batch_size 64 --item_page_size 0 --embed config/embed/glove.yaml
```

#### [PLMNR-NAML](https://arxiv.org/abs/2104.07413)

```bash
python trainer.py 
  --data config/data/mind-lm.yaml 
  --model config/model/bert-naml.yaml 
  --hidden_size 256 
  --lr 0.0001 
  --batch_size 64 
  --item_page_size 128  # set it as large as possible based on your GPU memory  
  --embed config/embed/bert.yaml  # generate the yaml first, by running python embed.py --model bertbase
  --use_lora 1 
  --lora_r 8 
  --lora_alpha 128 
  --lm bert  # indicate the language model
  --tune_from -2  # freeze the first N-1 layers, and tune the last layer, it is the same as --tune_from 10
```

#### [ONCE-DIRE-LLAMA1-NAML](https://arxiv.org/abs/2305.06566)

```bash
python trainer.py 
  --data config/data/mind-lm-prompt.yaml  # for more powerful language models, we suggest to use the data concatenated with natural prompts
  --model config/model/llama-naml.yaml 
  --hidden_size 256 
  --lr 0.0001 
  --batch_size 64 
  --item_page_size 64 
  --embed config/embed/llama.yaml  # generate the yaml first, by running python embed.py --model llama1
  --use_lora 1 
  --lora_r 32 
  --lora_alpha 128 
  --lm llama1 
  --tune_from -2  # freeze the first N-1 layers, and tune the last layer, it is the same as --tune_from 30
```

More documentations will be available soon.

## Updates

- **2025-02-18**: Legommenders v2.0, with multiple LLMs support, simplified configuration, more CTR predictors, and recbench-based datasets is released!
- **2025-01-06**: Legommenders v2.0 beta is released!
- **2024-12-05**: LSTUR model is now re-added to the Legommenders package, which was not compatible from Jan. 2024.
- **2024-01-23**: Legommenders partially supports the flatten sequential recommendation model. New models are added, including: MaskNet, GDCN, etc.
- **2023-10-16**: We clean the code and convert names of the item-side parameters.
- **2023-10-05**: The first recommender system package, Legommenders, with a modular-design is released!
- **2022-10-22**: Legommenders project is initiated.

## Citations

Legommenders have served as a fundamental framework for several research projects, including [ONCE](https://arxiv.org/abs/2305.06566), [SPAR](https://arxiv.org/abs/2402.10555),[GreenRec](https://arxiv.org/abs/2403.04736), and [UIST](https://arxiv.org/abs/2403.08206).
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

## Acknowledgement

We would like to thank Jieming Zhu and his [FuxiCTR](https://github.com/reczoo/FuxiCTR) project for providing multiple useful CTR predictors.

We would like to thank [transformers](https://huggingface.co/transformers/) for providing the pre-trained language models.

We would like to thank [UniTok V4](https://unitok.qijiong.work/) for providing the unified data tokenization service.

We would like to thank [RecBench](https://github.com/RecBench) for providing unified recommendation dataset preprocessing framework.

We would like to thank [Oba](https://pypi.org/project/oba/), [RefConfig](https://pypi.org/project/refconfig/), and [SmartDict](https://pypi.org/project/smartdict/) for providing useful tools for our project.