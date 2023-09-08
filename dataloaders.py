from typing import List, Tuple
import json


def load_corpus(datadir: str, split: str) -> Tuple[List[str],List[str]]:
    path_to_tsv = f"{datadir}/{split}"
    sentences, labels = [], []
    with open(path_to_tsv, "r", encoding="utf-8") as fp:
        rows = [line.strip().split("\t") for line in fp.readlines()]
        for row in rows:
            if len(row) == 2:
                sentence, label = row[0], row[1]
                sentences.append(sentence)
                labels.append(label)
    return sentences, labels


def load_enhanced_corpus(datadir: str, split: str) -> Tuple[List[str],List[str], List[str]]:
    sentences, contexts, labels = [], [], []
    with open(f"{datadir}/{split}", "r", encoding="utf-8") as fp:
        corpus = json.load(fp)
        for x in corpus["data"]:
            sentences.append(x["sentence"])
            context = x["generated_label"].strip().replace('"', "").replace("/", " ").replace("_", " ")
            contexts.append(context)
            labels.append(x["label"])
    return sentences, contexts, labels