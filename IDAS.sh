#!/usr/bin/env bash

for s in 1 2 3 4 5
do
    python3 label_generation.py \
    --dataset "stackoverflow"\
    --topk 8 \
    --n_prototypes 20 \
    --encoder "mtp" \
    --outfile "mtp_topk=8_prototypes=20_run${s}"
#   --permutation "mtp_topk=8_prototypes=20_run${s}" \ [OPTIONAL]
#      used for (1) ablations where the sample order needs to match the order in which labels were generated for the main results (with default hyperparams),
#      or (2) for a warm start, i.e., continuation of an previously started job where only a subset of the samples's labels were already generated.
done

