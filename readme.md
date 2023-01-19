# GreenNRL

Green News Representation Learning Framework

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

|            | NAML            | LSTUR            | NRMS            | DCN            | DIN            | BST            |
|------------|-----------------|------------------|-----------------|----------------|----------------|----------------|
| id-based   | id-NAML         | id-LSTUR         | id-NRMS         | DCN            | DIN            | BST            |
| text-based | NAML            | LSTUR            | NRMS            | text-DCN       | text-DIN       | text-BST       |
| bert-token | bert-token-NAML | bert-token-LSTUR | bert-token-NRMS | bert-token-DCN | bert-token-DIN | bert-token-BST |
| PLMNR      | PLMNR-NAML      | PLMNR-LSTUR      | PLMNR-NRMS      | PLMNR-DCN      | PLMNR-DIN      | PLMNR-BST      |
| bert-news  | bert-news-NAML  | bert-news-LSTUR  | bert-news-NRMS  | bert-news-DCN  | bert-news-DIN  | bert-news-BST  |
| mft-news   | mft-news-NAML   | mft-news-LSTUR   | mft-news-NRMS   | mft-news-DCN   | mft-news-DIN   | mft-news-BST   |

## Training and Testing

```bash
python worker_v2.py 
    --config config_v2/data/mind.yaml 
    --model config_v2/model/nrms.yaml 
    --exp config_v2/exp/tt-nrms.yaml
    --embed config_v2/embed/null.yaml
    --version small-v2 
```
