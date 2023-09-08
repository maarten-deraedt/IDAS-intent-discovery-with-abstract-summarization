#!/usr/bin/env bash


# MAIN RESULTS
python3 cluster.py \
  --dataset "banking77" \
  --configuration "mtp_topk=8_prototypes=77" \
  --encoder "all-mpnet-base-v2" \
  --outfile "my_results_all-mpnet-base-v2_topk=8_prototypes=77" \
  --max_smooth 45

python3 cluster.py \
  --dataset "stackoverflow" \
  --configuration "mtp_topk=8_prototypes=20" \
  --encoder "all-mpnet-base-v2" \
  --outfile "my_results_all-mpnet-base-v2_topk=8_prototypes=20" \
  --max_smooth 45


python3 cluster.py \
  --dataset "clinc150" \
  --configuration "paraphrase-mpnet-base-v2_topk=8_prototypes=150" \
  --encoder "all-mpnet-base-v2" \
  --outfile "my_results_all-mpnet-base-v2_topk=8_prototypes=150" \
  --max_smooth 45
