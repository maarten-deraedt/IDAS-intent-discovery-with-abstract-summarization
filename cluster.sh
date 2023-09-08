#!/usr/bin/env bash


# MAIN RESULTS
python3 cluster.py \
  --dataset "stackoverflow" \
  --configuration "mtp_topk=8_prototypes=20" \
  --encoder "mtp" \
  --outfile "my_results_mtp_topk=8_prototypes=20" \
  --max_smooth 45

python3 cluster.py \
  --dataset "banking77" \
  --configuration "mtp_topk=8_prototypes=77" \
  --encoder "mtp" \
  --outfile "my_results_mtp_topk=8_prototypes=77" \
  --max_smooth 45

python3 cluster.py \
  --dataset "clinc150" \
  --configuration "paraphrase-mpnet-base-v2_topk=8_prototypes=150" \
  --encoder "paraphrase-mpnet-base-v2" \
  --outfile "my_results_paraphrase-mpnet-base-v2_topk=8_prototypes=150" \
  --max_smooth 45



# ABLATIONS: uncomment to run the ablations on StackOverflow
#python3 cluster.py \
#  --dataset "stackoverflow" \
#  --configuration "mtp_topk=0_prototypes=20" \
#  --encoder "mtp" \
#  --outfile "my_results_mtp_topk=0_prototypes=20" \
#  --max_smooth 45
#
#python3 cluster.py \
#  --dataset "stackoverflow" \
#  --configuration "mtp_topk=1_prototypes=20" \
#  --encoder "mtp" \
#  --outfile "my_results_mtp_topk=1_prototypes=20" \
#  --max_smooth 45
#
#python3 cluster.py \
#  --dataset "stackoverflow" \
#  --configuration "mtp_topk=2_prototypes=20" \
#  --encoder "mtp" \
#  --outfile "my_results_mtp_topk=2_prototypes=20" \
#  --max_smooth 45
#
#python3 cluster.py \
#  --dataset "stackoverflow" \
#  --configuration "mtp_topk=4_prototypes=20" \
#  --encoder "mtp" \
#  --outfile "my_results_mtp_topk=4_prototypes=20" \
#  --max_smooth 45
#
#python3 cluster.py \
#  --dataset "stackoverflow" \
#  --configuration "mtp_topk=8_prototypes=40" \
#  --encoder "mtp" \
#  --outfile "my_results_mtp_topk=8_prototypes=40" \
#  --max_smooth 45
#
#python3 cluster.py \
#  --dataset "stackoverflow" \
#  --configuration "mtp_topk=16_prototypes=20" \
#  --encoder "mtp" \
#  --outfile "my_results_mtp_topk=16_prototypes=20" \
#  --max_smooth 45
#
#python3 cluster.py \
#  --dataset "stackoverflow" \
#  --configuration "random_mtp_topk=8_prototypes=20" \
#  --encoder "mtp" \
#  --outfile "my_results_random_mtp_topk=8_prototypes=20" \
#  --max_smooth 45