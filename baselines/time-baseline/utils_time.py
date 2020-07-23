import os
from collections import defaultdict
import anafora
from anafora import evaluate
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import InputFeatures
from spacy.lang.en import English


class TimeDataset(Dataset):

    def __init__(self, features, doc_indices=None):
        self.features = features
        self.doc_indices = doc_indices
        self.prediction = None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        return self.features[i]


class TimeInputFeatures(InputFeatures):

    def __init__(self, input_ids, attention_mask, offset_mapping, label):
        super().__init__(input_ids=input_ids, attention_mask=attention_mask, label=label)
        self.offset_mapping = offset_mapping


def read_labels(path, bio_mode=True):
    labels = ["O"]
    with open(path) as labels_file:
        if bio_mode:
            labels.extend([BI + label for label in labels_file.read().split("\n") for BI in ["B-", "I-"]])
        else:
            labels.extend(labels_file.read().split("\n"))
    return labels


def init_nlp_pipeline():
    nlp = English()
    sentencizer = nlp.create_pipe("sentencizer")
    nlp.add_pipe(sentencizer)
    return nlp


def create_datasets(model, nlp, dataset_path, train=False):
    text_directory_files = anafora.walk(dataset_path, xml_name_regex=".*((?<![.].{3})|[.]txt)$")
    features = []
    doc_indices = []
    if train:
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
            doc_features = from_doc_to_features(model, nlp, text_file_path, anafora_file_path, train=True)
            features.extend(doc_features)
            doc_indices.append((text_doc_name, doc_index, len(features)))
    else:
        for text_files in text_directory_files:
            doc_index = len(features)
            text_subdir_path, text_doc_name, text_file_names = text_files
            if len(text_file_names) != 1:
                raise Exception("Wrong number of text files in %s" % text_subdir_path)
            text_file_path = os.path.join(dataset_path, text_subdir_path, text_file_names[0])
            doc_features = from_doc_to_features(model, nlp, text_file_path)
            features.extend(doc_features)
            doc_indices.append((text_doc_name, doc_index, len(features)))
    return TimeDataset(features, doc_indices)


def bio_annotation(model, annotation, prefix):
    if not model.bio_mode:
        return model.labels.index(annotation)
    else:
        return model.labels.index(prefix + annotation)


def inner_subword(input_data, sent_idx, token_idx):
    if token_idx <= 0:
        return False
    else:
        previous_word = input_data.token_to_word(sent_idx, token_idx - 1)
        current_word = input_data.token_to_word(sent_idx, token_idx)
        return previous_word == current_word


def from_doc_to_features(model, nlp, text_path, anafora_path=None, train=False):
    with open(text_path) as txt_file:
        text = txt_file.read()
    doc = nlp(text)

    annotations = None
    if train:
        data = anafora.AnaforaData.from_file(anafora_path)
        annotations = dict()
        for annotation in data.annotations:
            label = annotation.type
            for span in annotation.spans:
                start, end = span
                annotations[start] = (end, label)

    input_raw = [sent.text_with_ws for sent in doc.sents]
    input_data = model.tokenizer(input_raw, return_tensors="pt", max_length=model.max_seq_length,
                                 padding="max_length", truncation="longest_first", return_offsets_mapping=True)
    if train:
        negative_attention_mask = (~input_data["attention_mask"].byte()).true_divide(255).long()
        input_data["labels"] = negative_attention_mask.mul(model.label_pad_id)
        # Assign label_pad to </s> token
        sent_indices = torch.arange(input_data["labels"].shape[0])
        last_non_padded = [sent_indices, input_data["labels"].argmax(dim=1)]
        input_data["labels"][last_non_padded] = model.label_pad_id
        # Assign label_pad to <s> token
        input_data["labels"][:, 0] = model.label_pad_id

    features = []
    sent_offset = 0
    for sent_idx, input_ids in enumerate(input_data["input_ids"]):
        attention_mask = input_data["attention_mask"][sent_idx]
        offset_mapping = input_data["offset_mapping"][sent_idx]
        labels = None
        if train:
            labels = input_data["labels"][sent_idx]

        start_open = None
        for token_idx, offset in enumerate(offset_mapping):
            start, end = offset.numpy()
            if start == 0 and end == 0:
                continue
            start += sent_offset
            end += sent_offset
            offset_mapping[token_idx][0] = start
            offset_mapping[token_idx][1] = end
            if train:
                # The annotation my have trailing spaces. Check if the current token is included in the span.
                if start_open is not None and annotations[start_open][0] <= start:
                    start_open = None
                # If nothing goes wrong, add the token to the opened annotation or open a new one
                if start_open is not None and start in annotations:
                    raise Exception("Offsets don't match in %s (%s, %s)" % (text_path, start, end))
                elif start_open is not None and model.pad_labels and inner_subword(input_data, sent_idx, token_idx):
                    labels[token_idx] = model.label_pad_id
                elif start_open is not None:
                    labels[token_idx] = bio_annotation(model, annotations[start_open][1], "I-")
                elif start in annotations:
                    labels[token_idx] = bio_annotation(model, annotations[start][1], "B-")
                    start_open = start
                # Check if the annotation ends in this token and close it
                if start_open is not None and end == annotations[start_open][0]:
                    start_open = None
        sent_offset += len(input_raw[sent_idx])
        features.append(
            TimeInputFeatures(
                input_ids,
                attention_mask,
                offset_mapping,
                labels
            )
        )

    return features


def is_b_label(label_id, bio_mode=True):
    if bio_mode:
        return label_id % 2 != 0
    else:
        return label_id > 0


def to_anafora(model, labels, features, doc_name="dummy"):
    data = anafora.AnaforaData()
    for sent_labels, sent_features in zip(labels, features):
        # Remove padding and <s> </s>
        special_mask = model.tokenizer.get_special_tokens_mask(sent_features.input_ids,
                                                               already_has_special_tokens=True)
        non_specials = np.count_nonzero(np.array(special_mask) == 0)
        sent_labels = sent_labels[1: non_specials + 1]
        sent_offsets = sent_features.offset_mapping[1: non_specials + 1]

        # Get the positions where the label changes. Keep all B- in BIO mode.
        label_shifts = np.flatnonzero(np.diff(sent_labels, prepend=0))
        b_starts = np.argwhere(is_b_label(sent_labels)).flatten() if model.bio_mode else []
        label_shifts = np.unique(np.concatenate((label_shifts, b_starts)))
        non_padded_shifts = sent_labels[label_shifts] != model.label_pad_id
        label_shifts = label_shifts[non_padded_shifts]
        label_shifts = np.append(label_shifts, sent_labels.size - 1)

        for i in range(label_shifts.size - 1):
            label_start = label_shifts[i]
            label_id = sent_labels[label_start]
            if is_b_label(label_id, model.bio_mode):
                label_next = label_shifts[i + 1]
                label_next_id = sent_labels[label_next]
                # I- is always 1 position after B- in label vocabulary
                is_i_label = label_next_id - label_id == 1 and not is_b_label(label_next_id, model.bio_mode)
                label_end = label_shifts[i + 2] if is_i_label else label_next
                label_end -= 1  # end must be the token previous to the next shift
                anafora.AnaforaEntity()
                entity = anafora.AnaforaEntity()
                num_entities = len(data.xml.findall("annotations/entity"))
                entity.id = "%s@%s" % (num_entities, doc_name)
                span_start = sent_offsets[label_start][0]
                span_end = sent_offsets[label_end][1]
                entity.spans = ((span_start, span_end),)
                entity.type = model.labels[label_id].replace("B-", "")
                data.annotations.append(entity)

    return data


def prepare_prediction(prediction):
    prediction = np.argmax(prediction, axis=2)
    return prediction


def score_predictions(model, dataset, prediction):
    scores_type = evaluate.Scores
    all_scores = defaultdict(lambda: scores_type())
    for doc_index in dataset.doc_indices:
        doc_name, doc_start, doc_end = doc_index
        doc_features = dataset.features[doc_start:doc_end]
        doc_labels = prediction.label_ids[doc_start:doc_end]
        doc_predictions = prediction.predictions[doc_start:doc_end]
        doc_predictions = prepare_prediction(doc_predictions)
        reference_data = to_anafora(model, doc_labels, doc_features)
        predicted_data = to_anafora(model, doc_predictions, doc_features)
        doc_scores = evaluate.score_data(reference_data, predicted_data)
        for name, scores in doc_scores.items():
            all_scores[name].update(scores)
    return all_scores["*"]


def write_predictions(model, dataset, out_path):
    for doc_index in dataset.doc_indices:
        doc_name, doc_start, doc_end = doc_index
        doc_predictions = dataset.prediction[doc_start:doc_end]
        doc_features = dataset.features[doc_start:doc_end]
        data = to_anafora(model, doc_predictions, doc_features, doc_name)
        doc_path = os.path.join(out_path, doc_name)
        os.makedirs(doc_path, exist_ok=True)
        doc_path = os.path.join(doc_path, "%s.TimeNorm.system.xml" % doc_name)
        data.to_file(doc_path)
