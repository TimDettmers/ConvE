#!/usr/bin/env python3

# the paths do not exist (we do not want to encourage the usage of WN18) and the filenames are off, but if you put in the WN18 files
# and adjust the path you will generate WN18RR.

predicates_to_remove = [
    '_member_of_domain_topic',
    '_synset_domain_usage_of',
    '_instance_hyponym',
    '_hyponym',
    '_member_holonym',
    '_synset_domain_region_of',
    '_part_of'
]


def read_triples(path):
    triples = []
    with open(path, 'rt') as f:
        for line in f.readlines():
            s, p, o = line.split('\t')
            triples += [(s.strip(), p.strip(), o.strip())]
    return triples


def write_triples(triples, path):
    with open(path, 'wt') as f:
        for (s, p, o) in triples:
            f.write('{}\t{}\t{}\n'.format(s, p, o))

train_triples = read_triples('original/wordnet-mlj12-train.txt')
valid_triples = read_triples('original/wordnet-mlj12-valid.txt')
test_triples = read_triples('original/wordnet-mlj12-test.txt')

filtered_train_triples = [(s, p, o) for (s, p, o) in train_triples if p not in predicates_to_remove]
filtered_valid_triples = [(s, p, o) for (s, p, o) in valid_triples if p not in predicates_to_remove]
filtered_test_triples = [(s, p, o) for (s, p, o) in test_triples if p not in predicates_to_remove]

write_triples(filtered_train_triples, 'wn18-train.tsv')
write_triples(filtered_valid_triples, 'wn18-valid.tsv')
write_triples(filtered_test_triples, 'wn18-test.tsv')
