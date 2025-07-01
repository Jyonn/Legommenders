# May I ask which configuration is recommended if I am currently using a multilingual news dataset?

https://github.com/Jyonn/Legommenders/issues/23

| Date   | Event                                 |
|--------|---------------------------------------|
| Jul. 1 | Support training on the xMIND dataset |

## Overview

The xMIND dataset only provides the translated news articles. It honestly follows the user interactions provided by the MIND dataset.

Therefore, only the news articles should be further processed, while the user history and interaction data can be directly shared from the processed MIND dataset.

This requires you to have the MIND dataset already processed and registered in the Legommenders framework.

## Step 1: Upgrade

Please upgrade Legommenders and other required packages to the latest version:

```bash
cd /path/to/Legommenders
git pull
pip install unitok -U
```

**Note: We assume that you have already downloaded the MIND dataset and processed it.**

## Step 2: Download the xMIND dataset

```bash
mkdir xMIND
cd xMIND
wget https://raw.githubusercontent.com/andreeaiana/xMIND/refs/heads/main/download.py
python download.py --languages cmn
python download.py --languages fin
```

## Step 3: Register the xMIND dataset

```bash
cd /path/to/Legommenders
cat .data
```

```text
mind = /path/to/MIND
xmindcmn = ~/PycharmProjects/xMIND/xMIND/cmn
xmindfin = ~/PycharmProjects/xMIND/xMIND/fin
```

## Step 4: Preprocess the xMIND dataset

Due to the multilingual nature of the xMIND dataset, we remove the BERT tokenizer and GloVe tokenizer, but only keep the Llama1 tokenizer. You can add more tokenizers if you want in `processors/xmind_processor.py`.

```bash
cd /path/to/Legommenders
python process.py --data xmindcmn
python process.py --data xmindfin
```

## Step 5: Have a Look at the Training Configurations

```bash
cd /path/to/Legommenders
cat config/data/xmind.yaml
cat config/data/mind-lm.yaml  # they have the same structure
```

### Step 6: Training on the xMIND dataset

```bash
cd /path/to/Legommenders
python train.py 
  --data config/data/xmind.yaml 
  --lm llama1
  --lang cmn
  --model config/model/naml.yaml
  --batch_size 64
  --hidden_size 64
  --lr 0.001
```

```text
(torch) ~/PycharmProjects/Legommenders git:[master]
python trainer.py --data config/data/xmind.yaml --lm llama1 --lang cmn --model config/model/naml.yaml --batch_size 64 --hidden_size 64
[00:00:00] |Trainer| START TIME: 2025-07-01 17:12:07.551274
[00:00:00] |Trainer| SIGNATURE: TM2Aybyy
[00:00:00] |Trainer| BASE DIR: checkpoints/mind/NAML
[00:00:00] |Trainer| python trainer.py --data config/data/xmind.yaml --lm llama1 --lang cmn --model config/model/naml.yaml --batch_size 64 --hidden_size 64
[00:00:00] |GPU| MPS available: using mac M series GPU
You are using the default legacy behaviour of the <class 'transformers.models.llama.tokenization_llama_fast.LlamaTokenizerFast'>. This is expected, and simply means that the `legacy` (previous) behavior will be used so nothing changes for you. If you want to use the new behaviour, set `legacy=False`. This should only be set if you understand what it means, and thoroughly read the reason why this was added as explained in https://github.com/huggingface/transformers/pull/24565 - if you loaded a llama tokenizer from a GGUF file you can ignore this message.
[00:00:02] |LegoUT| load 1 filter caches on UniTok(data/mind/test, size=2658091)
[00:00:04] |LegoUT| load 2 filter caches on UniTok(data/mind/train, size=5152779)
[00:00:04] |LegoUT| load 1 filter caches on UniTok(data/mind/valid, size=570223)
[00:00:10] |LegoUT| store filter cache on UniTok(data/mind/test, size=2658091)
[00:00:10] |LegoUT| load 2 filter caches on UniTok(data/mind/test, size=2658091)
[00:00:10] |Manager| Filter history with lambda x: x in <test> phase, sample num: 2658091 -> 2658091
[00:00:10] |LegoUT| store filter cache on UniTok(data/mind/train, size=5152779)
[00:00:10] |LegoUT| load 3 filter caches on UniTok(data/mind/train, size=5152779)
[00:00:10] |Manager| Filter history with lambda x: x in <train> phase, sample num: 5152779 -> 5152779
[00:00:11] |LegoUT| store filter cache on UniTok(data/mind/valid, size=570223)
[00:00:11] |LegoUT| load 2 filter caches on UniTok(data/mind/valid, size=570223)
[00:00:11] |Manager| Filter history with lambda x: x in <dev> phase, sample num: 570223 -> 570223
[00:00:11] |Manager| Filter history with lambda x: x in <fast_eval> phase, sample num: 91935 -> 91935
[W] |Instance| It is recommended to declare tokenizers and vocabularies in a UniTok context, using `with UniTok() as ut:`
[W] |Instance| It is recommended to declare tokenizers and vocabularies in a UniTok context, using `with UniTok() as ut:`
[00:00:11] |Manager| Selected Item Encoder: CNNOperator
[00:00:11] |Manager| Selected User Encoder: AdaOperator
[00:00:11] |Manager| Selected Predictor: DotPredictor
[00:00:11] |LegoUT| store filter cache on UniTok(data/mind/train, size=5152779)
[00:00:11] |LegoUT| load 4 filter caches on UniTok(data/mind/train, size=5152779)
[00:00:11] |Manager| Filter click with lambda x: x == 1 in <train> phase, sample num: 5152779 -> 208238
[00:00:11] |EmbeddingHub| global transformation type: auto
[00:00:11] |EmbeddingHub| global transformation dropout: 0.0
[00:00:11] |EmbeddingHub| --- create vocab llama1 (32000, 64)
[00:00:11] |EmbeddingHub| --- create vocab category (18, 64)
[00:00:11] |Legommender| set llm cache: False
[00:00:13] |Manager| {'user_id': 49108, 'neg': [], 'history': [6892, 30442, 14244, 29197, 28697, 16459, 1609, 27047, 31724, 27552, 11721, 23870, 27079, 1570, 32951], 'item_id': 62543, 'click': 0, 'index': 0}
[00:00:13] |Manager| {
    "user_id": "int",
    "history": {
        "input_ids": {
            "title@llama1": "tensor([50, 30], dtype=torch.int64)",
            "category": "tensor([50, 1], dtype=torch.int64)"
        },
        "attention_mask": {
            "title@llama1": "tensor([50, 30], dtype=torch.int64)",
            "category": "tensor([50, 1], dtype=torch.int64)"
        }
    },
    "item_id": {
        "input_ids": {
            "title@llama1": "tensor([5, 30], dtype=torch.int64)",
            "category": "tensor([5, 1], dtype=torch.int64)"
        },
        "attention_mask": {
            "title@llama1": "tensor([5, 30], dtype=torch.int64)",
            "category": "tensor([5, 1], dtype=torch.int64)"
        }
    },
    "click": "int",
    "index": "int",
    "__clicks_mask__": "tensor([50], dtype=torch.int64)"
}
[00:00:13] |Trainer| use single lr: 0.001
[00:00:13] |Trainer| embedding_vocab_table.llama1.weight torch.Size([32000, 64])
[00:00:13] |Trainer| embedding_vocab_table.category.weight torch.Size([18, 64])
[00:00:13] |Trainer| item_op.cnn.weight torch.Size([64, 64, 3])
[00:00:13] |Trainer| item_op.cnn.bias torch.Size([64])
[00:00:13] |Trainer| item_op.linear.weight torch.Size([64, 64])
[00:00:13] |Trainer| item_op.linear.bias torch.Size([64])
[00:00:13] |Trainer| item_op.additive_attention.encoder.0.weight torch.Size([256, 64])
[00:00:13] |Trainer| item_op.additive_attention.encoder.0.bias torch.Size([256])
[00:00:13] |Trainer| item_op.additive_attention.encoder.2.weight torch.Size([1, 256])
[00:00:13] |Trainer| user_op.additive_attention.encoder.0.weight torch.Size([256, 64])
[00:00:13] |Trainer| user_op.additive_attention.encoder.0.bias torch.Size([256])
[00:00:13] |Trainer| user_op.additive_attention.encoder.2.weight torch.Size([1, 256])
[00:05:32] |Trainer| [epoch 0] GAUC 0.7119                            
[00:05:32] |Trainer| save model to checkpoints/mind/NAML/TM2Aybyy.pt
[00:10:45] |Trainer| [epoch 1] GAUC 0.7118                           
[00:16:19] |Trainer| [epoch 2] GAUC 0.7172                            
[00:16:19] |Trainer| save model to checkpoints/mind/NAML/TM2Aybyy.pt
[00:22:17] |Trainer| [epoch 3] GAUC 0.7133                            
[00:27:42] |Trainer| [epoch 4] GAUC 0.7241                            
[00:27:43] |Trainer| save model to checkpoints/mind/NAML/TM2Aybyy.pt
[00:33:04] |Trainer| [epoch 5] GAUC 0.7247                            
[00:33:04] |Trainer| save model to checkpoints/mind/NAML/TM2Aybyy.pt
[00:38:23] |Trainer| [epoch 6] GAUC 0.7239                
[00:43:46] |Trainer| [epoch 7] GAUC 0.7265                
[00:43:46] |Trainer| save model to checkpoints/mind/NAML/TM2Aybyy.pt
[00:49:10] |Trainer| [epoch 8] GAUC 0.7249                           
[00:54:18] |Trainer| [epoch 9] GAUC 0.7287                           
[00:54:18] |Trainer| save model to checkpoints/mind/NAML/TM2Aybyy.pt
[00:59:36] |Trainer| [epoch 10] GAUC 0.7281                            
[01:04:47] |Trainer| [epoch 11] GAUC 0.7285                            
[01:09:56] |Trainer| [epoch 12] GAUC 0.7302                            
[01:09:56] |Trainer| save model to checkpoints/mind/NAML/TM2Aybyy.pt
[01:15:08] |Trainer| [epoch 13] GAUC 0.7296                           
[01:20:20] |Trainer| [epoch 14] GAUC 0.7292                           
[01:25:33] |Trainer| [epoch 15] GAUC 0.7288                           
[01:31:04] |Trainer| [epoch 16] GAUC 0.7308                            
[01:31:04] |Trainer| save model to checkpoints/mind/NAML/TM2Aybyy.pt
[01:36:26] |Trainer| [epoch 17] GAUC 0.7312                           
[01:36:26] |Trainer| save model to checkpoints/mind/NAML/TM2Aybyy.pt
[01:41:45] |Trainer| [epoch 18] GAUC 0.7302                           
[01:47:06] |Trainer| [epoch 19] GAUC 0.7320                            
[01:47:06] |Trainer| save model to checkpoints/mind/NAML/TM2Aybyy.pt
[01:52:30] |Trainer| [epoch 20] GAUC 0.7317                           
[01:57:48] |Trainer| [epoch 21] GAUC 0.7330                           
[01:57:48] |Trainer| save model to checkpoints/mind/NAML/TM2Aybyy.pt
[02:03:04] |Trainer| [epoch 22] GAUC 0.7315                            
[02:08:17] |Trainer| [epoch 23] GAUC 0.7323                           
[02:13:31] |Trainer| [epoch 24] GAUC 0.7327                            
[02:20:06] |Trainer| [epoch 25] GAUC 0.7322                           
[02:25:23] |Trainer| [epoch 26] GAUC 0.7327                           
[02:25:23] |Trainer| Early stop
[02:25:23] |Trainer| Training Ended
[02:25:23] |Trainer| load model from checkpoints/mind/NAML/TM2Aybyy.pt
/Users/bytedance/PycharmProjects/Legommenders/utils/metrics.py:179: UserWarning: Following existing recommendation repositories, the implementation of MRR is not the same as the original one. To get the original MRR, use MRR0 instead.
  warnings.warn('Following existing recommendation repositories, '
[02:27:54] |Trainer| GAUC: 0.6441     
[02:27:54] |Trainer| MRR: 0.2651
[02:27:54] |Trainer| NDCG@1: 0.1879
[02:27:54] |Trainer| NDCG@5: 0.2949
[02:27:54] |Trainer| NDCG@10: 0.3564
```
