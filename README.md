# IDAS: Intent Discovery with Abstractive Summarization
This repository contains the datasets and implementation of the NLP4ConvAI@ACL 2023 paper
["IDAS: Intent Discovery with Abstractive Summarization"](https://aclanthology.org/2023.nlp4convai-1.7/) by Maarten De Raedt, Fréderic Godin, Thomas Demeester, and Chris Develder.

For any questions about the paper or code contact the first author at [maarten.deraedt@ugent.be](mailto:maarten.deraedt@ugent.be).
If you find this repository useful for your own work, consider citing our paper:

````
@inproceedings{de-raedt-etal-2023-idas,
    title = "{IDAS}: Intent Discovery with Abstractive Summarization",
    author = "De Raedt, Maarten  and
      Godin, Fr{\'e}deric  and
      Demeester, Thomas  and
      Develder, Chris",
    booktitle = "Proceedings of the 5th Workshop on NLP for Conversational AI (NLP4ConvAI 2023)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.nlp4convai-1.7",
    doi = "10.18653/v1/2023.nlp4convai-1.7",
    pages = "71--88",
    abstract = "Intent discovery is the task of inferring latent intents from a set of unlabeled utterances, and is a useful step towards the efficient creation of new conversational agents. We show that recent competitive methods in intent discovery can be outperformed by clustering utterances based on abstractive summaries, i.e., {``}labels{''}, that retain the core elements while removing non-essential information. We contribute the IDAS approach, which collects a set of descriptive utterance labels by prompting a Large Language Model, starting from a well-chosen seed set of prototypical utterances, to bootstrap an In-Context Learning procedure to generate labels for non-prototypical utterances. The utterances and their resulting noisy labels are then encoded by a frozen pre-trained encoder, and subsequently clustered to recover the latent intents. For the unsupervised task (without any intent labels) IDAS outperforms the state-of-the-art by up to +7.42{\%} in standard cluster metrics for the Banking, StackOverflow, and Transport datasets. For the semi-supervised task (with labels for a subset of intents) IDAS surpasses 2 recent methods on the CLINC benchmark without even using labeled data.",
}
````
## Table of Contents
- [Installation](#installation)
- [Experiments](#experiment)


### Installation
Install the requirements.
```bash
$ pip3 install -r requirements.txt
```

```bash
$ mkdir -p MTP/pretrained_models/banking77
$ mkdir -p MTP/pretrained_models/stackoverflow
```

Download the MTP encoders for Banking and StackOverflow as directed [here](https://github.com/fanolabs/NID_ACLARR2022#download-external-pretrained-models), move the ``config.json`` and ``pytorch_model.json`` under ``MTP/pretrained_models/banking77 -or stackoverflow``.
The directory and file structure should match the structure below.
```
IDAS-intent-discovery-with-abstract-summarization
└─── datasets/   
│   └──banking77/
│   └──clinc150/
│   └──stackoverflow/
└─── MTP/   
│   └──pretrained_models/
│       └──banking77/
│       └──stackoverflow/
│       clnn.py
│       contrastive.py
│       dataloader.py
│       init_parameter.py
│       model.py
│       mtp.py
│       tools.py
│       README.md
│   cluster.py
│   cluster.sh
│   cluster_smpnet.sh
│   encode.py
│   IDAS.sh
│   label_generation.sh
│   metrics.py
│   README.md
│   requirements.txt
```

### Experiments
The `datasets/` directory contains the GPT-3 (text-davinci-003) generated labels for each original test file. 
For each test set, there are 5 .json files with generated labels, each corresponding to a different sample order. 
For example, Banking77's generated labels can be found in the files: `mtp_topk=8_prototypes=77_run{i}.json`, 
which were created using the default hyperparameters mentioned in our paper, with the number of nearest neighbors `topk=8`
and prototypes set to the ground truth number of intents (77). For Stackoverflow, we additionally provide the generated labels for our ablations, i.e., 
obtained with different hyperparameters.

To reproduce the main results with our generated labels:
```bash
./cluster.sh
```
and for the results with a more powerful sentence encoder during the clustering step: 
```bash
./cluster_smpnet.sh
```
The results will be written to a .json file in the corresponding dataset directory. 
Depending on the CPU, running the clustering experiments (especially the smoothing step) may take a while.

If you want to generate labels for a (new) dataset, you can adapt the parameters and run:
```bash
./IDAS.sh
```
In order to generate the labels, you will have to set the variable ``KEY`` to your personal OpenAI Key in ``label_generation.py``.
Note that OpenAI may have updated their API since the last time this code was run. 
