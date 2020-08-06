import os
import argparse
import torch
from torch.utils.data import Dataset
from transformers import (
    InputFeatures,
    AutoConfig,
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments
)
from spacy.lang.en import English
import anafora


class TimexDataset(Dataset):

    def __init__(self, doc_indices, features):
        self.doc_indices = doc_indices
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        return self.features[i]


class TimexInputFeatures(InputFeatures):

    def __init__(self, input_ids, attention_mask, offset_mapping, label):
        super().__init__(input_ids=input_ids, attention_mask=attention_mask, label=label)
        self.offset_mapping = offset_mapping


class TimexModel:

    def __init__(self, model_name):
        self.config = AutoConfig.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, config=self.config, use_fast=True)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name, config=self.config)
        self.nlp = self.init_nlp_pipeline()

    @staticmethod
    def init_nlp_pipeline():
        nlp = English()
        sentencizer = nlp.create_pipe("sentencizer")
        nlp.add_pipe(sentencizer)
        return nlp

    def bio_annotation(self, prefix, annotation):
        return self.config.label2id[prefix + annotation]

    def from_sent_to_features(self, input_data, sent_idx, sent_offset, annotations):
        input_ids = input_data["input_ids"][sent_idx]
        attention_mask = input_data["attention_mask"][sent_idx]
        offset_mapping = input_data["offset_mapping"][sent_idx]
        labels = input_data["labels"][sent_idx]

        start_open = None
        for token_idx, offset in enumerate(offset_mapping):
            start, end = offset.numpy()
            if start == end:
                continue
            start += sent_offset
            end += sent_offset
            offset_mapping[token_idx][0] = start
            offset_mapping[token_idx][1] = end

            # The annotation my have trailing spaces. Check if the current token is included in the span.
            if start_open is not None and annotations[start_open][0] <= start:
                start_open = None
            # If nothing goes wrong, add the token to the opened annotation or open a new one
            if start_open is not None and start in annotations:
                start_open = None
            elif start_open is not None:
                labels[token_idx] = self.bio_annotation("I-", annotations[start_open][1])
            elif start in annotations:
                labels[token_idx] = self.bio_annotation("B-", annotations[start][1])
                start_open = start
            # Check if the annotation ends in this token and close it
            if start_open is not None and end == annotations[start_open][0]:
                start_open = None

        return TimexInputFeatures(
            input_ids,
            attention_mask,
            offset_mapping,
            labels
        )

    def from_doc_to_features(self, text_path, anafora_path):
        with open(text_path) as txt_file:
            text = txt_file.read()
        doc = self.nlp(text)

        data = anafora.AnaforaData.from_file(anafora_path)
        annotations = dict()
        for annotation in data.annotations:
            label = annotation.type
            for span in annotation.spans:
                start, end = span
                annotations[start] = (end, label)

        input_raw = [sent.text_with_ws for sent in doc.sents]
        input_data = self.tokenizer(input_raw, return_tensors="pt", padding="max_length",
                                    truncation="longest_first", return_offsets_mapping=True)

        # Initialize label sequence with 0. Use ignore index for padding tokens
        negative_attention_mask = (~input_data["attention_mask"].byte()).true_divide(255).long()
        input_data["labels"] = negative_attention_mask.mul(self.config.label_pad_id)
        # Assign label_pad to </s> token
        sent_indices = torch.arange(input_data["labels"].shape[0])
        last_non_padded = [sent_indices, input_data["labels"].argmax(dim=1)]
        input_data["labels"][last_non_padded] = self.config.label_pad_id
        # Assign label_pad to <s> token
        input_data["labels"][:, 0] = self.config.label_pad_id

        features = []
        sent_offset = 0
        for sent_idx, _ in enumerate(input_data["input_ids"]):
            timex_features = self.from_sent_to_features(input_data, sent_idx, sent_offset, annotations)
            features.append(timex_features)
            sent_offset += len(input_raw[sent_idx])
        return features, annotations

    def create_datasets(self, dataset_path):
        text_directory_files = anafora.walk(dataset_path, xml_name_regex=".*((?<![.].{3})|[.]txt)$")
        features = []
        doc_indices = []
        for text_files in text_directory_files:
            doc_index = len(features)
            text_subdir_path, text_doc_name, text_file_names = text_files
            if len(text_file_names) != 1:
                raise Exception("Wrong number of text files in %s" % text_subdir_path)
            anafora_path = os.path.join(dataset_path, text_subdir_path)
            anafora_directory_files = anafora.walk(anafora_path, xml_name_regex="[.]xml$")
            anafora_directory_files = list(anafora_directory_files)
            if len(anafora_directory_files) != 1:
                raise Exception("Wrong structure in %s" % anafora_path)
            anafora_subdir_path, anafora_doc_name, anafora_file_names = anafora_directory_files[0]
            if len(anafora_file_names) != 1:
                raise Exception("Wrong number of anafora files in %s" % anafora_subdir_path)
            text_file_path = os.path.join(dataset_path, text_subdir_path, text_file_names[0])
            anafora_file_path = os.path.join(anafora_path, anafora_subdir_path, anafora_file_names[0])
            doc_features, doc_annotations = self.from_doc_to_features(text_file_path, anafora_file_path)
            features.extend(doc_features)
            doc_indices.append((text_doc_name, doc_index, len(features)))
        return TimexDataset(doc_indices, features)

    @staticmethod
    def data_collator(features):
        batch = dict()
        batch["input_ids"] = torch.stack([f.input_ids for f in features])
        batch["attention_mask"] = torch.stack([f.attention_mask for f in features])
        batch["labels"] = torch.stack([f.label for f in features])
        return batch

    def train(self, dataset, save_path):
        trainer_args = TrainingArguments(save_path)
        trainer = Trainer(
            model=self.model,
            args=trainer_args,
            train_dataset=dataset,
            data_collator=self.data_collator
        )
        trainer.train()
        trainer.save_model()
        self.tokenizer.save_pretrained(save_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="""%(prog)s runs SEMEVAL-2021 temporal baseline.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-t", "--train", metavar="DIR", dest="train_dir",
                        help="The root of the training set directory tree containing raw text and of Anafora XML.")
    parser.add_argument("-s", "--save", metavar="DIR", dest="save_dir",
                        help="The directory to save the model and the log files.")
    args = parser.parse_args()
    train_path = args.train_dir
    save_path = args.save_dir

    timex_model = TimexModel("clulab/roberta-timex-semeval")
    train_dataset = timex_model.create_datasets(train_path)
    timex_model.train(train_dataset, save_path)
