#!/bin/bash
mkdir data
mkdir data/WN18RR
mkdir data/YAGO3-10
mkdir data/FB15k-237
mkdir saved_models
tar -xvf WN18RR.tar.gz -C data/WN18RR
tar -xvf YAGO3-10.tar.gz -C data/YAGO3-10
tar -xvf FB15k-237.tar.gz -C data/FB15k-237
python wrangle_KG.py WN18RR
python wrangle_KG.py YAGO3-10
python wrangle_KG.py FB15k-237
