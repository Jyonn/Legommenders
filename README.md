# <img src="assets/lego.png" alt="icon" style="vertical-align: middle; height: 32px;"> Legommenders v2

*A Modular Framework for Recommender Systems in the Era of LLMs*

## Installation

```bash
gh repo clone Jyonn/Legommenders
cd Legommenders
pip install -r requirements.txt
```

## Quick Start with Three Steps

### Data Preprocessing

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
python trainer.py --data config/data/mind-glove.yaml --model config/model/naml.yaml --hidden_size 256 --lr 0.001 --batch_size 64 --item_page_size 0 --embed config/embed/glove.yaml
```

#### [NRMS](https://aclanthology.org/D19-1671/)

```bash
python trainer.py --data config/data/mind-glove.yaml --model config/model/nrms.yaml --hidden_size 256 --lr 0.001 --batch_size 64 --item_page_size 0 --embed config/embed/glove.yaml
```

#### [PLMNR-NAML](https://arxiv.org/abs/2104.07413)

```bash
python trainer.py --data config/data/mind-lm.yaml --model config/model/bert-naml.yaml --hidden_size 256 --lr 0.0001 --batch_size 64 --item_page_size 128 --embed config/embed/bert.yaml --use_lora 1 --lora_r 8 --lora_alpha 128 --lm bert
```

#### [ONCE-DIRE-LLAMA1-NAML](https://arxiv.org/abs/2305.06566)

```bash
python trainer.py --data config/data/mind-lm-prompt.yaml --model config/model/llama-naml.yaml --hidden_size 256 --lr 0.0001 --batch_size 64 --item_page_size 64 --embed config/embed/llama.yaml --use_lora 1 --lora_r 32 --lora_alpha 128 --lm llama1
```

More documentations will be available soon.

## Updates

### Jan. 6, 2025

- Legommenders v2.0 beta is released!

### Dec. 5, 2024

- LSTUR model is now re-added to the Legommenders package, which was not compatible from Jan. 2024.
- LLMs can be used for item encoder.

### Jan. 23, 2024

- Legommenders partially supports the flatten sequential recommendation model.
- New models are added, including: MaskNet, GDCN, etc.

### Oct. 16, 2023

- We clean the code and convert names of the item-side parameters. 

### Oct. 5, 2023

- The first recommender system package, Legommenders, with a modular-design is released!
- Legommenders involves a set of recommender system algorithms, including:
    - Matching based methods: NAML, NRMS, LSTUR, etc.
    - Ranking based methods: DCN, DeepFM, PNN, etc.


## Citations

Legommenders have served as a fundamental framework for several research projects, including [ONCE](https://arxiv.org/abs/2305.06566), [SPAR](https://arxiv.org/abs/2402.10555),[GreenRec](https://arxiv.org/abs/2403.04736), and [UIST](https://arxiv.org/abs/2403.08206).
If you find Legommenders useful in your research, please consider citing our project:

```
@article{legommenders,
  title={Legommenders: A Comprehensive Content-Based Recommendation Library with LLM Support},
  author={Liu, Qijiong and Fan, Lu and Wu, Xiao-Ming},
  journal={arXiv preprint arXiv:2412.15973},
  year={2024}
}
```
