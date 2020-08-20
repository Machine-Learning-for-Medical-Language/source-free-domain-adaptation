import logging
import os
import argparse

import numpy as np
from torch.utils.data.dataset import Dataset
from transformers.data.processors.utils import InputExample, InputFeatures
from transformers.data.processors.glue import glue_convert_examples_to_features
from transformers.data.processors.utils import DataProcessor
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

labels = ["-1", "1"]
max_length = 128
logger = logging.getLogger(__name__)


def predict(data_file, output_dir):

    # load the Huggingface config, tokenizer, and model
    model_name = "tmills/roberta_sfda_sharpseed"
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              config=config)
    model = AutoModelForSequenceClassification.from_pretrained(model_name,
                                                               config=config)

    # create a torch dataset from a tsv file
    test_dataset = NegationDataset.from_tsv(data_file, tokenizer)

    trainer = Trainer(
        model=model,
        args=TrainingArguments('save_run/'),
        compute_metrics=None,
    )

    predictions = trainer.predict(test_dataset=test_dataset).predictions
    predictions = np.argmax(predictions, axis=1)
    os.makedirs(output_dir, exist_ok=True)
    output_test_file = os.path.join(output_dir, 'system.tsv')

    with open(output_test_file, "w") as writer:
        logger.info("***** Test results *****")
        for index, item in enumerate(predictions):
            item = test_dataset.get_labels()[item]
            writer.write("%s\n" % item)


class NegationDataset(Dataset):
    def __init__(self, features):
        self.features = features
        self.label_list = ["-1", "1"]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]

    def get_labels(self):
        return self.label_list

    @classmethod
    def from_tsv(cls, tsv_file, tokenizer):
        """Creates examples for the test set."""
        lines = DataProcessor._read_tsv(tsv_file)
        examples = []
        for (i, line) in enumerate(lines):
            guid = 'instance-%d' % i
            if line[0] in labels:
                text_a = '\t'.join(line[1:])
            else:
                text_a = '\t'.join(line)

            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=None))

        features = glue_convert_examples_to_features(
            examples,
            tokenizer,
            max_length=max_length,
            label_list=labels,
            output_mode='classification',
        )
        return cls(features)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""%(prog)s runs SEMEVAL-2021 negation baseline.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-f", "--data_file", metavar="FILE", dest="data_file",
                        help="The input data file. Should be a .tsv with one instance per line.")
    parser.add_argument("-o", "--output_dir", metavar="DIR", dest="output_dir",
                        help="The output directory where the model predictions and will be written.")
    args = parser.parse_args()
    predict(args.data_file, args.output_dir)
