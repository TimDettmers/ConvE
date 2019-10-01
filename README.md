# ConvE
Convolutional 2D Knowledge Graph Embeddings resources.

Paper: [Convolutional 2D Knowledge Graph Embeddings](https://arxiv.org/abs/1707.01476)

Used in the paper, but *do not use these datasets for your research*:
[FB15k and WN18](https://everest.hds.utc.fr/doku.php?id=en:transe). Please also note that the Kinship and Nations datasets have a high number of inverse relationships which makes them unsuitable for research. Nations has +95% inverse relationships and Kinship about 48%.

## ConvE key facts

### Predictive performance

Dataset | MR | MRR | Hits@10 | Hits@3 | Hits@1
:--- | :---: | :---: | :---: | :---: | :---:
FB15k | 64 | 0.75 | 0.87 | 0.80 | 0.67
WN18 | 504 | 0.94 | 0.96 | 0.95 | 0.94
FB15k-237 | 246 | 0.32 | 0.49 | 0.35 | 0.24
WN18RR | 4766 | 0.43 | 0.51 | 0.44 | 0.39
YAGO3-10 | 2792 | 0.52 | 0.66 | 0.56 | 0.45
Nations | 2 | 0.82 | 1.00 | 0.88 | 0.72
UMLS | 1 | 0.94 | 0.99 | 0.97 | 0.92
Kinship | 2 | 0.83 | 0.98 | 0.91 | 0.73

### Run time performance

For an embedding size of 200 and batch size 128, a single batch takes on a GTX Titan X (Maxwell):
- 64ms for 100,000 entities
- 80ms for 1,000,000 entities

### Parameter efficiency


Parameters | ConvE/DistMult MRR | ConvE/DistMult Hits@10 | ConvE/DistMult Hits@1
:--- | :---: | :---: | :---:
~5.0M | 0.32 / 0.24 |  0.49 / 0.42 | 0.24 / 0.16
1.89M | 0.32 / 0.23 | 0.49 / 0.41 | 0.23 / 0.15
0.95M| 0.30 / 0.22 | 0.46 / 0.39 | 0.22 / 0.14
0.24M | 0.26 / 0.16 | 0.39 / 0.31 | 0.19 / 0.09

ConvE with 8 times less parameters is still more powerful than DistMult. Relational Graph Convolutional Networks use roughly 32x more parameters to have the same performance as ConvE.

## Installation

This repo supports Linux and Python installation via Anaconda. 

1. Install [PyTorch](https://github.com/pytorch/pytorch) using [Anaconda](https://www.continuum.io/downloads).
2. Install the requirements `pip install -r requirements.txt`
3. Download the default English model used by [spaCy](https://github.com/explosion/spaCy), which is installed in the previous step `python -m spacy download en`
4. Run the preprocessing script for WN18RR, FB15k-237, YAGO3-10, UMLS, Kinship, and Nations: `sh preprocess.sh`
5. You can now run the model

## Running a model

Parameters need to be specified by white-space tuples for example:
```
CUDA_VISIBLE_DEVICES=0 python main.py --model conve --data FB15k-237 \
                                      --input-drop 0.2 --hidden-drop 0.3 --feat-drop 0.2 \
                                      --lr 0.003 --preprocess
```
will run a ConvE model on FB15k-237.

To run a model, you first need to preprocess the data once. This can be done by specifying the `--preprocess` parameter:
```
CUDA_VISIBLE_DEVICES=0 python main.py --data DATASET_NAME --preprocess
```
After the dataset is preprocessed it will be saved to disk and this parameter can be omitted.
```
CUDA_VISIBLE_DEVICES=0 python main.py --data DATASET_NAME
```
The following parameters can be used for the `--model` parameter:
```
conve
distmult
complex
```

The following datasets can be used for the `--data` parameter:
```
FB15k-237
WN18RR
YAGO3-10
umls
kinship
nations
```

And here a complete list of parameters.
```
Link prediction for knowledge graphs

optional arguments:
  -h, --help            show this help message and exit
  --batch-size BATCH_SIZE
                        input batch size for training (default: 128)
  --test-batch-size TEST_BATCH_SIZE
                        input batch size for testing/validation (default: 128)
  --epochs EPOCHS       number of epochs to train (default: 1000)
  --lr LR               learning rate (default: 0.003)
  --seed S              random seed (default: 17)
  --log-interval LOG_INTERVAL
                        how many batches to wait before logging training
                        status
  --data DATA           Dataset to use: {FB15k-237, YAGO3-10, WN18RR, umls,
                        nations, kinship}, default: FB15k-237
  --l2 L2               Weight decay value to use in the optimizer. Default:
                        0.0
  --model MODEL         Choose from: {conve, distmult, complex}
  --embedding-dim EMBEDDING_DIM
                        The embedding dimension (1D). Default: 200
  --embedding-shape1 EMBEDDING_SHAPE1
                        The first dimension of the reshaped 2D embedding. The
                        second dimension is infered. Default: 20
  --hidden-drop HIDDEN_DROP
                        Dropout for the hidden layer. Default: 0.3.
  --input-drop INPUT_DROP
                        Dropout for the input embeddings. Default: 0.2.
  --feat-drop FEAT_DROP
                        Dropout for the convolutional features. Default: 0.2.
  --lr-decay LR_DECAY   Decay the learning rate by this factor every epoch.
                        Default: 0.995
  --loader-threads LOADER_THREADS
                        How many loader threads to use for the batch loaders.
                        Default: 4
  --preprocess          Preprocess the dataset. Needs to be executed only
                        once. Default: 4
  --resume              Resume a model.
  --use-bias            Use a bias in the convolutional layer. Default: True
  --label-smoothing LABEL_SMOOTHING
                        Label smoothing value to use. Default: 0.1
  --hidden-size HIDDEN_SIZE
                        The side of the hidden layer. The required size
                        changes with the size of the embeddings. Default: 9728
                        (embedding size 200).
```
To reproduce most of the results in the ConvE paper, you can use the default parameters and execute the command below:
```
CUDA_VISIBLE_DEVICES=0 python main.py --data DATASET_NAME
```
For the reverse model, you can run the provided file with the name of the dataset name and a threshold probability:

```
python inverse_model.py WN18RR 0.9
```

### Changing the embedding size for ConvE

If you want to change the embedding size you can do that via the ``--embedding-dim` parameter. However, for ConvE, since the embedding is reshaped as a 2D embedding one also needs to pass the first dimension of the reshaped embedding (`--embedding-shape1`) while the second dimension is infered. When once changes the embedding size, the hidden layer size `--hidden-size` also needs to be different but it is difficult to determine before run time. The easiest way to determine the hidden size is to run the model, let it run on an error due to wrong shape, and then reshape according to the dimension in the error message.

Example: Change embedding size to be 100. We want 10x10 2D embeddings. We run `python main.py --embedding-dim 100 --embedding-shape1 10` and we run on an error due to wrong hidden dimension:
```python
   ret = torch.addmm(bias, input, weight.t())
RuntimeError: size mismatch, m1: [128 x 4608], m2: [9728 x 100] at /opt/conda/conda-bld/pytorch_1565272271120/work/aten/src/THC/generic/THCTensorMathBlas.cu:273
```

Now we change the hidden dimension to 4608 accordingly: `python main.py --embedding-dim 100 --embedding-shape1 10 --hidden-size 4608`. Now the model runs with an embedding size of 100 and 10x10 2D embeddings.

### Adding new datasets

To run it on a new datasets, copy your dataset folder into the data folder and make sure your dataset split files have the name `train.txt`, `valid.txt`, and `test.txt` which contain tab separated triples of a knowledge graph. Then execute `python wrangle_KG.py FOLDER_NAME`, afterwards, you can use the folder name of your dataset in the dataset parameter.

### Adding your own model

You can easily write your own knowledge graph model by extending the barebone model `MyModel` that can be found in the `model.py` file.

### Quirks

There are some quirks of this framework.
1. The model currently ignores data that does not fit into the specified batch size, for example if your batch size is 100 and your test data is 220, then 20 samples will be ignored. This is designed in that way to improve performance on small datasets. To test on the full test-data you can save the model checkpoint, load the model (with the `--resume True` variable) and then evaluate with a batch size that fits the test data (for 220 you could use a batch size of 110). Another solution is to just use a fitting batch size from the start, that is, you could train with a batch size of 110.

### Issues

It has been noted that [#6](/../../issues/6) WN18RR does contain 212 entities in the test set that do not appear in the training set. About 6.7% of the test set is affected. This means that most models will find it impossible to make any reasonable predictions for these entities. This will make WN18RR appear more difficult than it really is, but it should not affect the usefulness of the dataset. If all researchers compared to the same datasets the scores will still be comparable.

## Logs

Some log files of the original research are included in the repo (logs.tar.gz). These log files are mostly unstructured in names and might be created from checkpoints so that it is difficult to comprehend them. Nevertheless, it might help to replicate the results or study the behavior of the training under certain conditions and thus I included them here.

## Citation

If you found this codebase or our work useful please cite us:
```
@inproceedings{dettmers2018conve,
	Author = {Dettmers, Tim and Pasquale, Minervini and Pontus, Stenetorp and Riedel, Sebastian},
	Booktitle = {Proceedings of the 32th AAAI Conference on Artificial Intelligence},
	Title = {Convolutional 2D Knowledge Graph Embeddings},
	Url = {https://arxiv.org/abs/1707.01476},
	Year = {2018},
        pages  = {1811--1818},
  	Month = {February}
}



```
