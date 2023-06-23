# GNRS: Green News Recommender System

## Environment

```bash
pip install -r requirements.txt
```

## Data Processing

_Please specify the path to the data in python file_

```bash
cd process/mind
python processor.py
```

## Configuration

### Data

Please refer to `config_v2/data/mind.yaml` for the data configuration.

### Model

We support the following models on both MIND small and large datasets:

|            | NAML       | LSTUR       | NRMS       | DCN       | DIN           | BST           |
|------------|------------|-------------|------------|-----------|---------------|---------------|
| id-based   | id-NAML    | id-LSTUR    | id-NRMS    | DCN       | DIN           | BST           |
| text-based | NAML       | LSTUR       | NRMS       | text-DCN  | text-DIN      | text-BST      |
| PLMNR      | PLMNR-NAML | PLMNR-LSTUR | PLMNR-NRMS | PLMNR-DCN | PLMNR-DIN     | PLMNR-BST     |
| BERT       | BERT-NAML  | BERT-LSTUR  | BERT-NRMS  | BERT-DCN  | bert-news-DIN | bert-news-BST |
| MFT        | MFT-NAML   | MFT-LSTUR   | MFT-NRMS   | MFT-DCN   | MFT-DIN       | MFT-BST       |

## Training and Testing

```bash
python worker.py 
    --config config/data/mind.yaml 
    --model config/model/nrms.yaml 
    --exp config/exp/tt-nrms.yaml
    --embed config/embed/null.yaml
    --version small-v2 
```
