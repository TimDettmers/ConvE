# ConvE
Convolutional 2D Knowledge Graph Embeddings resources. Source code coming soon.

Paper: [2D Convolutional Graph Embeddings](https://arxiv.org/abs/1707.01476)

Used in the paper, but *do not use these datasets for your research*:
[FB15k and WN18](https://everest.hds.utc.fr/doku.php?id=en:transe)

## Installation

This repo supports Linux and Python installation via Anaconda. 

1. Install [PyTorch](https://github.com/pytorch/pytorch) using [Anaconda](https://www.continuum.io/downloads)
2. Install the requirements `pip install -r requirements`
3. Run the preprocessing script for WN18RR, FB15k-237, and YAGO3-10: `sh preprocess.sh`
4. You can now run the model

## Running a model

This is a bit messy, but it works.

1. Select your model and dataset in the main.py script
2. You can pass common parameters like learning rate (lr), dropout values (input_drop, hidden_drop, feat_drop) as parameters to the script like so:
```
CUDA_VISIBLE_DEVICES=0 python main.py input_drop 0.2 hidden_drop 0.3 feat_drop 0.2 lr 0.003
```
The command above in conjunction with ConvE as a model will yield state-of-the-art results for most link prediction datasets.

For the reverse model, you can run the provided file with the name of the dataset name and a threshold probability:

```
python reverse_rule.py WN18RR 0.9
```

To run it on new datasets, either adjust the path structure in the file, or copy your dataset folder into the data folder and make sure your dataset split files have the name `train.txt`, `valid.txt`, and `test.txt` which contain tab separated triples of a knowledge graph.
