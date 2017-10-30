from __future__ import print_function
from os.path import join
import json

import argparse
import datetime
import requests
import json
import urllib
import pickle
import os
import numpy as np
import operator
import sys

rdm = np.random.RandomState(234234)

if len(sys.argv) > 1:
    dataset_name = sys.argv[1]
else:
    dataset_name = 'FB15k-237'
    #dataset_name = 'FB15k'
    #dataset_name = 'yago'
    #dataset_name = 'WN18RR'

print('Processing dataset {0}'.format(dataset_name))

rdm = np.random.RandomState(2342423)
base_path = 'data/{0}/'.format(dataset_name)
files = ['train.txt', 'valid.txt', 'test.txt']

data = []
for p in files:
    with open(join(base_path, p)) as f:
        data = f.readlines() + data


def convert_mid(e):
    if e in mid2data:
        if 'name' in mid2data[e]:
            return mid2data[e]['name']
    return e


egraph = {}
d_egraph = {}
d_egraph_sets = {}
test_cases = {}
e_rel_direction ={}
for p in files:
    test_cases[p] = []
    d_egraph_sets[p] = {}


for p in files:
    with open(join(base_path, p)) as f:
        for i, line in enumerate(f):
            e1, rel, e2 = line.strip().split('\t')
            e1 = e1.strip()
            e2 = e2.strip()
            rel = rel.strip()

            if (e1 , rel) not in d_egraph:
                d_egraph[(e1, rel)] = set()

#            if (e2,  rel) not in d_egraph:
#                d_egraph[(e2, rel)] = set()

            if (e1,  rel) not in d_egraph_sets[p]:
                d_egraph_sets[p][(e1, rel)] = set()
            #if (e2, rel) not in d_egraph_sets[p]:
                #d_egraph_sets[p][(e2, rel)] = set()

            if e1+rel not in e_rel_direction:
                e_rel_direction[e1+rel] = 'left'
            else:
                e_rel_direction[e1+rel] = 'bidirectional'

#            if e2+rel not in e_rel_direction:
#                e_rel_direction[e2+rel] = 'right'
#            else:
#                e_rel_direction[e2+rel] == 'bidirectional'
#
            d_egraph[(e1, rel)].add(e2)
            #d_egraph[(e2, rel)].add(e1)
            test_cases[p].append([e1, rel, e2])
            d_egraph_sets[p][(e1, rel)].add(e2)
            #d_egraph_sets[p][(e2, rel)].add(e1)


#print('largest entities relations:')
#for i in range(10):
#    print(sorted_x[i])

def write_e1rel_graph(cases, graph, path):
    with open(path, 'w') as f:
        n = len(graph)
        for i, key in enumerate(graph):
            e1, rel = key
            entities = list(graph[key])
            direction = e_rel_direction[e1+rel]

            entities1 = " ".join(entities)

            data_point = {}
            data_point['e1'] = e1
            data_point['e2'] = str(rdm.choice(entities))
            data_point['rel'] = rel
            data_point['direction1'] = direction
            data_point['direction2'] = 'none'
            data_point['e2_multi1'] =  entities1
            data_point['e2_multi2'] = "None"

            f.write(json.dumps(data_point)  + '\n')

def write_e1rel_ranking_graph(cases, graph, path):
    with open(path, 'w') as f:
        n = len(cases)
        for i, (e1, rel, e2) in enumerate(cases):
            entities1 = list(graph[(e1, rel)])
            entities2 = list(graph[(e2, rel)])
            direction1 = e_rel_direction[e1+rel]
            direction2 = e_rel_direction[e2+rel]

            entities1 = " ".join(entities1)
            entities2 = " ".join(entities2)

            data_point = {}
            data_point['e1'] = e1
            data_point['e2'] = e2
            data_point['rel'] = rel
            data_point['direction1'] = direction1
            data_point['direction2'] = direction2
            data_point['e2_multi1'] = entities1
            data_point['e2_multi2'] = entities2

            f.write(json.dumps(data_point)  + '\n')


all_cases = test_cases['train.txt'] + test_cases['valid.txt'] + test_cases['test.txt']
write_e1rel_graph(test_cases['train.txt'], d_egraph_sets['train.txt'], 'data/{0}/e1rel_to_e2_train.json'.format(dataset_name))
write_e1rel_ranking_graph(test_cases['valid.txt'], d_egraph, join('data/{0}/e1rel_to_e2_ranking_dev.json'.format(dataset_name)))
write_e1rel_ranking_graph(test_cases['test.txt'], d_egraph, 'data/{0}/e1rel_to_e2_ranking_test.json'.format(dataset_name))
write_e1rel_graph(all_cases, d_egraph, 'data/{0}/e1rel_to_e2_full.json'.format(dataset_name))
