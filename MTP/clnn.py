"""
main program for running internal pre-training and CLNN

some functions are modified from
https://github.com/thuiar/DeepAligned-Clustering/blob/main/DeepAligned.py
"""

from MTP.model import CLBert
from MTP.init_parameter import init_model
from MTP.dataloader import Data
from MTP.mtp import InternalPretrainModelManager
from MTP.tools import *
from dataloaders import load_corpus
from MTP.dataloader import convert_examples_to_features
import argparse

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class CLNNModelManager:
    def __init__(self, args, labels, pretrained_model=None):
        set_seed(args.seed)
        n_gpu = torch.cuda.device_count()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_labels = len(set(labels))
        self.model = CLBert(args.bert_model, device=self.device)

        if n_gpu > 1:
            self.model = nn.DataParallel(self.model)

        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    def encode(self, sentences, feat_dim: int = 768, max_seq_length: int = 50):
        features = []
        for ex_index, example in enumerate(sentences):
            tokens_a = self.tokenizer.tokenize(example)

            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
            # (b) For single sequences:
            #  tokens:   [CLS] the dog is hairy . [SEP]
            #  type_ids: 0   0   0   0  0     0 0
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambigiously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.
            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            segment_ids = [0] * len(tokens)
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            features.append({
                "input_ids": input_ids,
                "input_mask": input_mask,
                "segment_ids": segment_ids}
            )

        print(f"Successfully obtained tokenized sentence representations for {len(sentences)} sentences!")

        self.model.eval()
        total_features = torch.empty((0,feat_dim)).to(self.device)

        for sample in tqdm(features, desc="Extracting representations"):
            # batch = tuple(t.to(self.device) for t in batch)
            # input_ids, input_mask, segment_ids, label_ids = batch
            input_ids = torch.tensor([sample['input_ids']], dtype=torch.long)
            input_mask = torch.tensor([sample['input_mask']], dtype=torch.long)
            segment_ids = torch.tensor([sample['segment_ids']], dtype=torch.long)
            X = {"input_ids": input_ids, "attention_mask": input_mask, "token_type_ids": segment_ids}
            with torch.no_grad():
                feature = self.model(X, output_hidden_states=True)["hidden_states"]

            total_features = torch.cat((total_features, feature))

        return total_features.numpy()


def mtp(sentences, labels, pretrained_model: str, max_seq_len: int = 55):
    # print('Data and Parameters Initialization...')
    args = argparse.Namespace(
        bert_model=f'./MTP/pretrained_models/{pretrained_model}',
        disable_pretrain="True",
        feat_dim=768,
        grad_clip=1,
        known_cls_ration=0.0,
        labeled_ratio=0.1,
        lr=1e-05,
        lr_pre=5e-5,
        method="CLNN",
        num_pretrain_epochs=100,
        num_train_epochs=34,
        pretrain_batch_size=64,
        report_pretrain=False,
        rtr_prob=0.25,
        save_model_path=None,
        save_results_path='\'clnn_outputs\'',
        seed=0,
        temp=0.07,
        tokenizer='bert-base-uncased',
        topk=50,
        train_batch_size=128,
        update_per_epoch=5,
        view_strategy='\'rtr\'',
        wait_patient=20,
        warmup_proportion=0.1,
    )
    # print(args)

    # sentences, labels = load_corpus(datadir, split)
    encoder = CLNNModelManager(args, labels)
    embeddings = encoder.encode(sentences=sentences, max_seq_length=max_seq_len)

    return embeddings