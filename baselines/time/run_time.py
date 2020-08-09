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


def predict(predict_dir, output_dir):

    # load the Huggingface config, tokenizer, and model
    model_name = "clulab/roberta-timex-semeval"
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              config=config,
                                              use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(model_name,
                                                            config=config)

    # load the spacy sentence segmenter
    nlp = English()
    nlp.add_pipe(nlp.create_pipe("sentencizer"))

    # create a torch dataset from a directory of text files
    dataset = TimexDataset.from_texts(predict_dir, nlp, tokenizer)

    # get predictions from the torch model
    trainer = Trainer(
        model=model,
        args=TrainingArguments("save_run/"),
        data_collator=lambda features: dict(
            input_ids=torch.stack([f.input_ids for f in features]),
            attention_mask=torch.stack([f.attention_mask for f in features]))
    )
    predictions, _, _ = trainer.predict(dataset)

    # write the predictions in Anafora XML format
    write_anafora(output_dir, dataset, predictions, tokenizer, config)


class TimexInputFeatures(InputFeatures):

    def __init__(self, input_ids, attention_mask, offset_mapping):
        super().__init__(input_ids=input_ids, attention_mask=attention_mask)
        self.offset_mapping = offset_mapping

    @classmethod
    def from_sentence(cls, input_data, sent_idx, sent_offset):
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
        return cls(
            input_ids,
            attention_mask,
            offset_mapping
        )


class TimexDataset(Dataset):

    def __init__(self, doc_indices, features):
        self.doc_indices = doc_indices
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        return self.features[i]

    @classmethod
    def from_texts(cls, text_dir, nlp, tokenizer):
        if not os.path.exists(text_dir):
            raise Exception("The %s directory does not exist." % text_dir)
        text_directory_files = anafora.walk(text_dir, xml_name_regex=".*((?<![.].{3})|[.]txt)$")
        features = []
        doc_indices = []
        for text_files in text_directory_files:
            doc_index = len(features)
            text_subdir_path, text_doc_name, text_file_names = text_files
            if len(text_file_names) != 1:
                raise Exception("Wrong number of text files in %s" % text_subdir_path)
            text_file_path = os.path.join(text_dir, text_subdir_path, text_file_names[0])
            with open(text_file_path) as txt_file:
                text = txt_file.read()
            doc = nlp(text)
            input_raw = [sent.text_with_ws for sent in doc.sents]
            input_data = tokenizer(input_raw,
                                   return_tensors="pt",
                                   padding="max_length",
                                   truncation="longest_first",
                                   return_offsets_mapping=True)
            sent_offset = 0
            for sent_idx, _ in enumerate(input_data["input_ids"]):
                features.append(TimexInputFeatures.from_sentence(
                    input_data,
                    sent_idx,
                    sent_offset))
                sent_offset += len(input_raw[sent_idx])
            doc_indices.append((text_subdir_path, doc_index, len(features)))
        return cls(doc_indices, features)


def write_anafora(output_dir, dataset, predictions, tokenizer, config):

    def add_entity(data, doc_name, label, offset):
        entity_label = config.id2label[label] if label > 0 else None
        if entity_label is not None:
            anafora.AnaforaEntity()
            entity = anafora.AnaforaEntity()
            num_entities = len(data.xml.findall("annotations/entity"))
            entity.id = "%s@%s" % (num_entities, doc_name)
            entity.spans = ((offset[0], offset[1]),)
            entity.type = entity_label.replace("B-", "")
            data.annotations.append(entity)

    for doc_index in dataset.doc_indices:
        doc_subdir, doc_start, doc_end = doc_index
        doc_name = os.path.basename(doc_subdir)
        doc_features = dataset.features[doc_start:doc_end]
        doc_predictions = predictions[doc_start:doc_end]
        doc_predictions = np.argmax(doc_predictions, axis=2)
        data = anafora.AnaforaData()
        for sent_labels, sent_features in zip(doc_predictions, doc_features):
            # Remove padding and <s> </s>
            special_mask = tokenizer.get_special_tokens_mask(sent_features.input_ids,
                                                             already_has_special_tokens=True)
            non_specials = np.count_nonzero(np.array(special_mask) == 0)
            sent_labels = sent_labels[1: non_specials + 1]
            sent_offsets = sent_features.offset_mapping[1: non_specials + 1]

            previous_label = 0
            previous_offset = [None, None]  # (start, end)
            for token_label, token_offset in zip(sent_labels, sent_offsets):
                label_diff = token_label - previous_label
                if token_label % 2 != 0:  # If odd number, it is B label
                    add_entity(data, doc_name, previous_label, previous_offset)
                    previous_label = token_label
                    previous_offset = token_offset
                elif label_diff == 1:  # If even number and diff with previous is 1, it is I label
                    previous_offset[1] = token_offset[1]
                elif previous_label > 0:  # If current is O label and previous not O we must write it.
                    add_entity(data, doc_name, previous_label, previous_offset)
                    previous_label = 0
                    previous_offset = [None, None]
            if previous_label > 0:  # If remaining previous not O we must write it.
                entity_label = config.id2label[previous_label]
                add_entity(data, doc_name, entity_label, previous_offset)
        doc_path = os.path.join(output_dir, doc_subdir)
        os.makedirs(doc_path, exist_ok=True)
        doc_path = os.path.join(doc_path,
                                "%s.TimeNorm.system.completed.xml" % doc_name)
        data.to_file(doc_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="""%(prog)s runs SEMEVAL-2021 temporal baseline.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-p", "--predict", metavar="DIR", dest="predict_dir",
                        help="The root of the directory tree containing raw text for prediction.")
    parser.add_argument("-o", "--output", metavar="DIR", dest="output_dir",
                        help="The directory to store the prediction in Anafora XML.")
    args = parser.parse_args()
    predict(args.predict_dir, args.output_dir)
