# <img src="assets/lego.png" alt="icon" style="vertical-align: middle; height: 32px;"> Legommenders

*A modular framework for recommender systems*

## Installation

```bash
gh repo clone Jyonn/Legommenders
cd Legommenders
pip install -r requirements.txt  # Note: Legommenders is not compatible to the latest version of transformers yet if you want to finetune LLaMA-based models.
```

## Updates

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
@online{legommenders,
  author = {Liu, Qijiong},
  title = {Legommenders: A Modular Framework for Recommender Systems},
  year = {2023},
  url = {https://github.com/Jyonn/Legommenders}
}
```
