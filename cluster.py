import json

from sklearn.cluster import KMeans
from dataloaders import load_enhanced_corpus
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score

from encode import sentence_transformer
from metrics import clustering_score
from MTP.clnn import mtp
from argparse import ArgumentParser

import numpy as np


# required for MTP encoder.
MAX_SEQ_LENGTHS = {'stackoverflow': 45, 'banking77': 55, "transport": 45}


# print latest results as intermediate outputs.
def print_summary(results, topk=None):
    if topk is not None:
        ari = results["smoothed"][topk]["aris"][-1]
        nmi = results["smoothed"][topk]["nmis"][-1]
        acc = results["smoothed"][topk]["accs"][-1]
        sil = results["smoothed"][topk]["sils"][-1]
        print(f"[{topk}-smoothing]  {round(ari,2)} & {round(nmi, 2)} & {round(acc, 2)} & sil: {round(sil, 2)}")
    else:
        print("              ARI  & ACC  & NMI ")
        for key in ["sentence", "label", "average"]:
            ari = results[key]["aris"][-1]
            nmi = results[key]["nmis"][-1]
            acc = results[key]["accs"][-1]
            print(f"[{key}]  {round(ari, 2)} & {round(nmi, 2)} & {round(acc, 2)}")


def neighbors(similarity_matrix, sample_index, topk: int):
    return np.argsort(similarity_matrix[sample_index, :])[0:topk + 1]


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="stackoverflow",
                        help="The name of dataset directory for which we wish to generate labels.")
    parser.add_argument("--configuration", type=str,
                        help="the configuration for which we perform the clustering.")
    parser.add_argument("--encoder", type=str, default="mtp")
    parser.add_argument("--max_smooth", type=int, default=45,
                        help="maximum of number of smoothing neighbors")
    parser.add_argument("--num_runs", type=int, default=10,
                        help="the number of times each configuration is clustered")
    parser.add_argument("--outfile", type=str,
                        help="The output file where the generated labels are written to (same directory as the dataset).")
    args = parser.parse_args()
    print(args)

    outfile = f"datasets/{args.dataset}/{args.outfile}.json"
    max_smooth = args.max_smooth
    encoder_name = args.encoder
    configuration = args.configuration
    dataset = args.dataset
    num_runs = args.num_runs

    files = [f"{configuration}_run{i}.json" for i in [1, 2, 3, 4, 5]] # each configuration should have 5 independent runs.
    results = {
        "sentence": {
            "aris": [],
            "nmis": [],
            "accs": [],
            "aris_sd": [],
            "nmis_sd": [],
            "accs_sd": [],
        },
        "label": {
            "aris": [],
            "nmis": [],
            "accs": [],
            "aris_sd": [],
            "nmis_sd": [],
            "accs_sd": [],
        },
        "average": {
            "aris": [],
            "nmis": [],
            "accs": [],
            "aris_sd": [],
            "nmis_sd": [],
            "accs_sd": [],
        },
        "smoothed": dict()
    }

    random_lower, random_upper = 99999, 9999999 # for large random seeds
    topks = [i for i in range(5, max_smooth+1)]
    for file in files:
        sentences, contexts, labels = load_enhanced_corpus(f"datasets/{dataset}", f"{file}")
        print(f"Starting to cluster for: {file}.")

        labels = LabelEncoder().fit_transform(labels)
        if encoder_name == "mtp":
            print(f"Using encoder {encoder_name}")
            embeddings = mtp(sentences, labels, pretrained_model=dataset, max_seq_len=MAX_SEQ_LENGTHS[dataset])
            context_embeddings = mtp(contexts, labels, pretrained_model=dataset, max_seq_len=MAX_SEQ_LENGTHS[dataset])
        elif encoder_name == "mtp_test":
            print(f"Using encoder {encoder_name}")
            embeddings = mtp(sentences, labels, pretrained_model=f"{dataset}-test", max_seq_len=MAX_SEQ_LENGTHS[dataset])
            context_embeddings = mtp(contexts, labels, pretrained_model=f"{dataset}-test", max_seq_len=MAX_SEQ_LENGTHS[dataset])
        else:
            print(f"Using encoder {encoder_name}")
            embeddings = sentence_transformer(sentences, model_name=encoder_name)
            context_embeddings = sentence_transformer(contexts, model_name=encoder_name)

        dimension = embeddings.shape[1]
        average_embeddings = (embeddings + context_embeddings) / 2.0

        # perform  k-means for the utterance only, label only, and average encoding strategies.
        aris, accs, nmis = [], [], []
        aris_context, accs_context, nmis_context = [], [], []
        aris_avg, accs_avg, nmis_avg = [], [], []
        k = len(set(labels))
        for i in range(num_runs):
            clusterer = KMeans(n_clusters=k, n_init=10)
            clusterer.fit(embeddings)
            labels_pred = clusterer.labels_
            scores = clustering_score(np.array(labels), np.array(labels_pred))

            clusterer = KMeans(n_clusters=k, n_init=10)
            clusterer.fit(context_embeddings)
            labels_pred = clusterer.labels_
            scores_context = clustering_score(np.array(labels), np.array(labels_pred))

            clusterer = KMeans(n_clusters=k, n_init=10)
            clusterer.fit(average_embeddings)
            labels_pred = clusterer.labels_
            scores_avg = clustering_score(np.array(labels), np.array(labels_pred))

            aris.append(scores['ARI'])
            accs.append(scores['ACC'])
            nmis.append(scores['NMI'])

            aris_context.append(scores_context['ARI'])
            accs_context.append(scores_context['ACC'])
            nmis_context.append(scores_context['NMI'])

            aris_avg.append(scores_avg['ARI'])
            accs_avg.append(scores_avg['ACC'])
            nmis_avg.append(scores_avg['NMI'])


        results["sentence"]["aris"].append(float(np.mean(aris)))
        results["sentence"]["aris_sd"].append(float(np.std(aris)))
        results["sentence"]["nmis"].append(float(np.mean(nmis)))
        results["sentence"]["nmis_sd"].append(float(np.std(nmis)))
        results["sentence"]["accs"].append(float(np.mean(accs)))
        results["sentence"]["accs_sd"].append(float(np.std(accs)))

        results["label"]["aris"].append(float(np.mean(aris_context)))
        results["label"]["aris_sd"].append(float(np.std(aris_context)))
        results["label"]["nmis"].append(float(np.mean(nmis_context)))
        results["label"]["nmis_sd"].append(float(np.std(nmis_context)))
        results["label"]["accs"].append(float(np.mean(accs_context)))
        results["label"]["accs_sd"].append(float(np.std(accs_context)))

        results["average"]["aris"].append(float(np.mean(aris_avg)))
        results["average"]["aris_sd"].append(float(np.std(aris_avg)))
        results["average"]["nmis"].append(float(np.mean(nmis_avg)))
        results["average"]["nmis_sd"].append(float(np.std(nmis_avg)))
        results["average"]["accs"].append(float(np.mean(accs_avg)))
        results["average"]["accs_sd"].append(float(np.std(accs_avg)))

        print_summary(results)

        # Select the number of smoothing neighbors that optimizes the silhouette score.
        for topk in topks:
            smoothed_embeddings = average_embeddings
            similarities = cosine_distances(smoothed_embeddings, smoothed_embeddings)
            new_embeddings = []
            for i in range(similarities.shape[0]):
                neighborhood = neighbors(similarities, i, topk=topk)
                new_embedding = np.zeros(shape=(dimension,))
                for neighbor in neighborhood.tolist():
                    new_embedding += smoothed_embeddings[neighbor, :]
                new_embeddings.append(new_embedding / neighborhood.size)
            smoothed_embeddings = np.array(new_embeddings)

            aris_smoothed, accs_smoothed, nmis_smoothed, silhouettes = [], [], [], []
            for i in range(num_runs):
                clusterer = KMeans(n_clusters=k, n_init=10)
                clusterer.fit(smoothed_embeddings)
                labels_pred = clusterer.labels_
                scores_smoothed = clustering_score(np.array(labels), np.array(labels_pred))
                silhouette = silhouette_score(smoothed_embeddings, labels_pred)

                aris_smoothed.append(scores_smoothed['ARI'])
                accs_smoothed.append(scores_smoothed['ACC'])
                nmis_smoothed.append(scores_smoothed['NMI'])
                silhouettes.append(silhouette)

            if topk not in results["smoothed"].keys():
                results["smoothed"][topk] = {
                    "aris": [float(np.mean(aris_smoothed))],
                    "nmis": [float(np.mean(nmis_smoothed))],
                    "accs": [float(np.mean(accs_smoothed))],
                    "aris_sd": [float(np.std(aris_smoothed))],
                    "nmis_sd": [float(np.std(nmis_smoothed))],
                    "accs_sd": [float(np.std(accs_smoothed))],
                    "sils": [float(np.mean(silhouettes))]
                }
            else:
                results["smoothed"][topk]["aris"].append(float(np.mean(aris_smoothed)))
                results["smoothed"][topk]["nmis"].append(float(np.mean(nmis_smoothed)))
                results["smoothed"][topk]["accs"].append(float(np.mean(accs_smoothed)))
                results["smoothed"][topk]["aris_sd"].append(float(np.std(aris_smoothed)))
                results["smoothed"][topk]["nmis_sd"].append(float(np.std(nmis_smoothed)))
                results["smoothed"][topk]["accs_sd"].append(float(np.std(accs_smoothed)))
                results["smoothed"][topk]["sils"].append(float(np.mean(silhouettes)))
            print_summary(results, topk)

    with open(outfile, "w") as fp:
        json.dump(results, fp, indent=4)