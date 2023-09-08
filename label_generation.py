from encode import sentence_transformer
from dataloaders import load_corpus
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from argparse import ArgumentParser
from MTP.clnn import mtp

import sys
import json
import time
import traceback
import openai
import numpy as np
import multiprocessing, timeit


def send_request(prompt, queue):
    response = openai.Completion.create(model="text-davinci-003", prompt=prompt, temperature=0, max_tokens=250)
    queue.put(response)


def limited_wait(s, queue):
    start = timeit.default_timer()
    while timeit.default_timer() - start < s and queue.empty():
        continue
    return not queue.empty()


def instruct(prompt, max_tries=10, wait_time=10):
    for i in range(0, max_tries):
        queue = multiprocessing.Queue()
        p = multiprocessing.Process(target=send_request, args=(prompt, queue,))
        p.start()

        if limited_wait(wait_time, queue):
            return queue.get()
        else:
            print("Trying again...")
            p.terminate()
    return (None, max_tries*wait_time)


MAX_SEQ_LENGTHS = {'stackoverflow': 45, 'banking77': 55, "transport": 45} # required for MTP encoder.
URL = "https://api.openai.com/v1/completions"
KEY = "ADD YOUR KEY HERE"
openai.api_key = KEY
HEADERS = {'Content-Type': 'application/json', 'Authorization': f'Bearer {KEY}'}
DATA = {
    "model": "text-davinci-003",
    "temperature": 0.0,
    "max_tokens": 250,
    "n": 1,
    "top_p": 1,
}
INSTRUCTIONS = {
    "clinc150": {
        "prototype": "Describe the chatbot question in a maximum of 5 words.\n",
        "in-context": "Classify the question into of the provided chatbot labels.\n",
    },
    "banking77": {
        "prototype": "Describe the banking question in a maximum of 5 words.\n",
        "in-context": "Classify the question into of the provided banking labels.\n"
    },
    "stackoverflow": {
        "prototype": "Identify the technology in question.\n",
        "in-context": "Classify the question into one of the provided technologies.\n"
    }
}


if __name__ == '__main__':
    print('Parameter initialization.')
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str,
                        help="The name of dataset directory for which we wish to generate labels.")
    parser.add_argument("--n_prototypes", type=int,
                        help="The number of initial clusters to retrieve prototypes.")
    parser.add_argument("--encoder", type=str, default="mtp")
    parser.add_argument("--topk", type=int, default=8,
                        help="The number of ICL-demonstrations.")
    parser.add_argument("--permutation", type=str,
                        help="path to json outfile with generated labels parameters, to run the ablations in the same order.")
    parser.add_argument("--outfile", type=str,
                        help="The output file where the generated labels are written to (same directory as the dataset).")
    args = parser.parse_args()
    print(args)

    dataset = f"datasets/{args.dataset}/test.tsv"
    outfile = f"datasets/{args.dataset}/{args.outfile}.json"
    topk = args.topk
    n_prototypes = args.n_prototypes
    encoder_name = args.encoder

    try:
        with open(outfile, "r") as fp:
            label_augmented = json.load(fp)
        print(f"Found existing {outfile},  proceeding with warm start.")
        index = len(label_augmented["data"])
        warm_start = True
        if "usage" in label_augmented.keys():
            usage = label_augmented["usage"]
        permutation = np.array(label_augmented["permutation"])
        prototypes, prototype_labels, prototype_noisy_labels = [], [], []
        for prototype in label_augmented["data"]:
            prototypes.append(prototype["sentence"])
            prototype_labels.append(prototype["label"])
            prototype_noisy_labels.append(prototype["generated_label"])
        if encoder_name == "mtp":
            print(f"Using encoder: {encoder_name}")
            prototype_embeddings = mtp(prototypes, prototype_labels, pretrained_model=args.dataset,
                                       max_seq_len=MAX_SEQ_LENGTHS[args.dataset])
        elif encoder_name == "mtp_test":
            print(f"Using encoder: {encoder_name}")
            prototype_embeddings = mtp(prototypes, prototype_labels, pretrained_model=f"{args.dataset}-test",
                                   max_seq_len=MAX_SEQ_LENGTHS[args.dataset])
        else:
            print(f"Using encoder: {encoder_name}")
            prototype_embeddings = sentence_transformer(prototypes, encoder_name)
        warm_start = True
    except Exception:
        warm_start = False
        print(traceback.print_exc())
        print("Could not find an existing file, starting from scratch.")
        label_augmented = {
            "data": []
        }
        index = 0

    sentences, labels = load_corpus(f"datasets/{args.dataset}", "test.tsv")
    if encoder_name == "mtp":
        print(f"Using encoder: {encoder_name}")
        sentence_embeddings = mtp(sentences, labels, pretrained_model=args.dataset,
                                  max_seq_len=MAX_SEQ_LENGTHS[args.dataset])
    else:
        print(f"Using encoder: {encoder_name}")
        sentence_embeddings = sentence_transformer(sentences, encoder_name)
    print(f"Encoded all sentences! There are {len(set(labels))} labels.")

    # shuffle the data since data order is the primary source of potential performance variation.
    if not warm_start:
        if args.permutation is not None:
            with open(f"datasets/{args.dataset}/{args.permutation}.json", "r") as fp:
                permutation = np.array(json.load(fp)["permutation"])
                print(f"Running ablation with provided permutation from {args.permutation}.")
        else:
            rng = np.random.default_rng()
            permutation = rng.permutation(np.arange(len(labels)))
            print("Randomly permuted the data!")
    print(permutation[:10])
    sentence_embeddings = sentence_embeddings[permutation, :]
    labels = [labels[i] for i in permutation]
    sentences = [sentences[i] for i in permutation]
    label_augmented["permutation"] = permutation.tolist()

    """
    Independently generate labels if no In-Context Learning is used, i.e., as an ablation.
    """
    if topk == 0:
        for i, sentence in enumerate(sentences):
            DATA["prompt"] = INSTRUCTIONS[args.dataset]["prototype"]
            if args.dataset == "stackoverflow":
                DATA["prompt"] += f'question: "{sentence}"\ntechnology:'
            else:
                DATA["prompt"] += f'question: "{sentence}"\nlabel:'
            try:
                response = instruct(prompt=DATA["prompt"])
                noisy_label = response["choices"][0]["text"].strip().replace('"', "")
                print(f"[Sentence ({i})] {sentence} [{labels[i]}] [{noisy_label}]")
                time.sleep(3)
            except Exception:
                print(f"Something went wrong with: {sentence}")
                print(traceback.format_exc())
                noisy_label = "ERROR"
            label_augmented["data"].append({
                "sentence": sentence,
                "label": labels[i],
                "generated_label": noisy_label,
                "original_index": int(permutation[i])
            })
            with open(outfile, "w") as fp:
                json.dump(label_augmented, fp, indent=4)
        sys.exit()

    """Retrieve prototypes"""
    if not warm_start:
        clusterer = KMeans(n_clusters=n_prototypes)
        clusters = clusterer.fit_predict(sentence_embeddings)
        print(f"Induced the clusters with k-means.")

        cluster_map = dict()
        for i, sentence in enumerate(sentences):
            label, cluster = labels[i], clusters[i]
            if cluster not in cluster_map.keys():
                cluster_map[cluster] = [{
                    "sentence": sentence,
                    "label": label,
                    "index": i
                }]
            else:
                cluster_map[cluster].append({
                    "sentence": sentence,
                    "label": label,
                    "index": i,
                })
        print(f"Created the cluster map with {len(cluster_map.keys())} prototypes.")

        prototypes, prototype_labels, prototype_noisy_labels, prototype_embeddings = [], [], [], []
        for i, cluster in enumerate(cluster_map.keys()):
            cluster_indices = [sample["index"] for sample in cluster_map[cluster]]
            cluster_embeddings = sentence_embeddings[cluster_indices, :]
            cluster_centroid = np.mean(sentence_embeddings[cluster_indices, :], axis=0)
            sims = cosine_similarity(np.array([cluster_centroid]), cluster_embeddings)
            prototype_index = np.flip(np.argsort(sims, axis=1)).flatten()[0]
            index = cluster_indices[prototype_index]

            """
            Independently generate the label of the prototype. 
            """
            prototype = sentences[index]
            DATA["prompt"] = INSTRUCTIONS[args.dataset]["prototype"]
            if args.dataset == "stackoverflow":
                DATA["prompt"] += f'question: "{prototype}"\ntechnology:'
            else:
                DATA["prompt"] += f'question: "{prototype}"\nlabel:'
            try:
                response = instruct(prompt=DATA["prompt"])
                noisy_label = response["choices"][0]["text"].strip().replace('"', "")
                prototypes.append(sentences[index])
                prototype_noisy_labels.append(noisy_label)
                prototype_labels.append(labels[index])
                prototype_embeddings.append(sentence_embeddings[index, :])
                print(f"[Prototype ({i})] {prototype} [{labels[index]}] [{noisy_label}]")
                time.sleep(3)
            except Exception:
                print(f"Something went wrong with: {prototype}")
                print(traceback.format_exc())
                noisy_label = "ERROR"
            label_augmented["data"].append({
                "sentence": prototype,
                "label": labels[index],
                "generated_label": noisy_label,
                "original_index": int(permutation[index])
            })

        prototype_embeddings = np.array(prototype_embeddings)
        print(f"Created the prototypes!")

    """
    For each non-prototype, retrieve its k nearest prototypes and use them as ICL-demonstrations to generate its label.
    """
    usage = 0 if not warm_start else usage
    for i, sentence in enumerate(sentences):
        if sentence not in prototypes:
            sims = cosine_similarity(np.array([sentence_embeddings[i, :]]), prototype_embeddings)
            nearest_prototype_indices = np.flip(np.argsort(sims, axis=1)).flatten()[0:topk]

            DATA["prompt"] = INSTRUCTIONS[args.dataset]["in-context"]
            demonstrations = ""
            for proto_index in nearest_prototype_indices:
                l = prototype_noisy_labels[proto_index]
                if args.dataset == "stackoverflow":
                    demonstrations += f'question: "{prototypes[proto_index]}"\ttechnology: "{l}"\n'
                else:
                    demonstrations += f'question: "{prototypes[proto_index]}"\tlabel: "{l}"\n'
            DATA["prompt"] += demonstrations
            if args.dataset == "stackoverflow":
                DATA["prompt"] += f'question: "{sentence}"\ttechnology:'
            else:
                DATA["prompt"] += f'question: "{sentence}"\tlabel:'

            try:
                response = instruct(prompt=DATA["prompt"])
                noisy_label = response["choices"][0]["text"].strip().replace('"', "")
                usage += int(response["usage"]["total_tokens"])
                prototypes.append(sentence)
                prototype_noisy_labels.append(noisy_label)
                prototype_embeddings = np.vstack((prototype_embeddings, sentence_embeddings[i, :]))
            except Exception:
                print(traceback.format_exc())
                print(f"Something went wrong with: {sentence}")
                noisy_label = "ERROR"
            label_augmented["data"].append(
                {
                    "original_index": int(permutation[i]),
                    "sentence": sentence,
                    "label": labels[i],
                    "generated_label": noisy_label,
                    "prompt": demonstrations,
                }
            )
            print(f'[({i}) usage:{round((usage / 1000) * 0.02,4)})$] {sentence} [{labels[i]}] [{noisy_label}]')
            label_augmented["usage"] = usage
            with open(outfile, "w") as fp:
                json.dump(label_augmented, fp, indent=4)
            time.sleep(3)