import os
import argparse
import numpy as np
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

    def __init__(self, input_ids, attention_mask, offset_mapping):
        super().__init__(input_ids=input_ids, attention_mask=attention_mask)
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

    def from_sent_to_features(self, input_data, sent_idx, sent_offset):
        input_ids = input_data["input_ids"][sent_idx]
        attention_mask = input_data["attention_mask"][sent_idx]
        offset_mapping = input_data["offset_mapping"][sent_idx]
        for token_idx, offset in enumerate(offset_mapping):
            start, end = offset.numpy()
            if start == end:
                continue
            start += sent_offset
            end += sent_offset
            offset_mapping[token_idx][0] = start
            offset_mapping[token_idx][1] = end
        return TimexInputFeatures(
            input_ids,
            attention_mask,
            offset_mapping
        )

    def from_doc_to_features(self, text_path):
        with open(text_path) as txt_file:
            text = txt_file.read()
        doc = self.nlp(text)
        input_raw = [sent.text_with_ws for sent in doc.sents]
        input_data = self.tokenizer(input_raw, return_tensors="pt", padding="max_length",
                                    truncation="longest_first", return_offsets_mapping=True)
        features = []
        sent_offset = 0
        for sent_idx, _ in enumerate(input_data["input_ids"]):
            timex_features = self.from_sent_to_features(input_data, sent_idx, sent_offset)
            features.append(timex_features)
            sent_offset += len(input_raw[sent_idx])
        return features

    def create_datasets(self, dataset_path):
        text_directory_files = anafora.walk(dataset_path, xml_name_regex=".*((?<![.].{3})|[.]txt)$")
        features = []
        doc_indices = []
        for text_files in text_directory_files:
            doc_index = len(features)
            text_subdir_path, text_doc_name, text_file_names = text_files
            if len(text_file_names) != 1:
                raise Exception("Wrong number of text files in %s" % text_subdir_path)
            text_file_path = os.path.join(dataset_path, text_subdir_path, text_file_names[0])
            doc_features = self.from_doc_to_features(text_file_path)
            features.extend(doc_features)
            doc_indices.append((text_doc_name, doc_index, len(features)))
        return TimexDataset(doc_indices, features)

    def add_entity(self, data, doc_name, label, offset):
        entity_label = self.config.id2label[label] if label > 0 else None
        if entity_label is not None:
            anafora.AnaforaEntity()
            entity = anafora.AnaforaEntity()
            num_entities = len(data.xml.findall("annotations/entity"))
            entity.id = "%s@%s" % (num_entities, doc_name)
            entity.spans = ((offset[0], offset[1]),)
            entity.type = entity_label.replace("B-", "")
            data.annotations.append(entity)

    def prediction_to_anafora(self, labels, features, doc_name):
        data = anafora.AnaforaData()
        for sent_labels, sent_features in zip(labels, features):
            # Remove padding and <s> </s>
            special_mask = self.tokenizer.get_special_tokens_mask(sent_features.input_ids,
                                                                  already_has_special_tokens=True)
            non_specials = np.count_nonzero(np.array(special_mask) == 0)
            sent_labels = sent_labels[1: non_specials + 1]
            sent_offsets = sent_features.offset_mapping[1: non_specials + 1]

            previous_label = 0
            previous_offset = [None, None]  # (start, end)
            for token_label, token_offset in zip(sent_labels, sent_offsets):
                label_diff = token_label - previous_label
                if token_label % 2 != 0:  # If odd number, it is B label
                    self.add_entity(data, doc_name, previous_label, previous_offset)
                    previous_label = token_label
                    previous_offset = token_offset
                elif label_diff == 1:  # If even number and diff with previous is 1, it is I label
                    previous_offset[1] = token_offset[1]
                elif previous_label > 0:  # If current is O label and previous not O we must write it.
                    self.add_entity(data, doc_name, previous_label, previous_offset)
                    previous_label = 0
                    previous_offset = [None, None]
            if previous_label > 0:  # If remaining previous not O we must write it.
                entity_label = self.config.id2label[previous_label]
                self.add_entity(data, doc_name, entity_label, previous_offset)

        return data

    def write_predictions(self, dataset, predictions, out_path):
        for doc_index in dataset.doc_indices:
            doc_name, doc_start, doc_end = doc_index
            doc_features = dataset.features[doc_start:doc_end]
            doc_predictions = predictions[doc_start:doc_end]
            doc_predictions = np.argmax(doc_predictions, axis=2)
            data = self.prediction_to_anafora(doc_predictions, doc_features, doc_name)
            doc_path = os.path.join(out_path, doc_name)
            os.makedirs(doc_path, exist_ok=True)
            doc_path = os.path.join(doc_path, "%s.TimeNorm.system.completed.xml" % doc_name)
            data.to_file(doc_path)

    @staticmethod
    def data_collator(features):
        batch = dict()
        batch["input_ids"] = torch.stack([f.input_ids for f in features])
        batch["attention_mask"] = torch.stack([f.attention_mask for f in features])
        return batch

    def predict(self, dataset):
        trainer_args = TrainingArguments("save_run/")
        trainer = Trainer(
            model=self.model,
            args=trainer_args,
            data_collator=self.data_collator
        )
        prediction, _, _ = trainer.predict(dataset)
        return prediction


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="""%(prog)s runs SEMEVAL-2021 temporal baseline.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-p", "--predict", metavar="DIR", dest="predict_dir",
                        help="The root of the directory tree containing raw text for prediction.")
    parser.add_argument("-o", "--output", metavar="DIR", dest="output_dir",
                        help="The directory to store the prediction in Anafora XML.")
    args = parser.parse_args()
    predict_path = args.predict_dir
    output_path = args.output_dir

    timex_model = TimexModel("clulab/roberta-timex-semeval")
    predict_dataset = timex_model.create_datasets(predict_path)
    prediction = timex_model.predict(predict_dataset)
    timex_model.write_predictions(predict_dataset, prediction, output_path)
