import logging
from os.path import join
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from torch.utils.data.dataset import Dataset
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction
from transformers.data.processors.utils import DataProcessor, InputExample, InputFeatures
from transformers.data.processors.glue import glue_convert_examples_to_features
from transformers.data.processors.utils import DataProcessor
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)

labels = ["-1", "1"]
logger = logging.getLogger(__name__)

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

def create_examples(lines):
    """Creates examples for the test set."""
    examples = []
    for (i, line) in enumerate(lines):
        # flip the signs so that 1 is negated, that way the f1 calculation is automatically
        # the f1 score for the negated label.
        guid='instance-%d' % (i)
        if line[0] in labels:
            text_a = '\t'.join(line[1:])
        else:
            text_a = '\t'.join(line)

        examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=None))
    return examples

@dataclass
class RunArguments:
    """
    Arguments pertaining to the running of our model
    """

    data_file: str = field(
        metadata={"help": "The input data file. Should be a .tsv with one instance per line"}
    )

    output_dir: str = field(
        metadata={"help": "The output directory where the model predictions and will be written."}
    )

    # label_file: str = field(
    #     metadata={"help": "The file where output (labels) will be written"}
    # )

    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    eval_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )
    device: str = field(
        default='cuda',
        metadata={"help": "Whether to use cuda or cpu device (default is cuda)"}
    )

def main():
    parser = HfArgumentParser((RunArguments,))
    run_args, = parser.parse_args_into_dataclasses()
    # A few args that the trainer wants
    run_args.seed = 42
    run_args.local_rank = -1
    run_args.fp16 = False
    run_args.dataloader_drop_last = False
    run_args.prediction_loss_only = False
    run_args.n_gpu = 1
    run_args.past_index = -1

    model_name = "tmills/roberta_sfda_sharpseed"

    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=2,
        finetuning_task='negation',
        cache_dir=run_args.cache_dir,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    examples = create_examples(DataProcessor._read_tsv(run_args.data_file))
    features = glue_convert_examples_to_features(
                    examples,
                    tokenizer,
                    max_length=run_args.max_seq_length,
                    label_list=labels,
                    output_mode='classification',
                )
    test_dataset = NegationDataset(features)

    trainer = Trainer(
        model=model,
        args=run_args,
        compute_metrics=None,
    )


    predictions = trainer.predict(test_dataset=test_dataset).predictions
    predictions = np.argmax(predictions, axis=1)
    output_test_file = join(run_args.output_dir, 'system.tsv')

    with open(output_test_file, "w") as writer:
        logger.info("***** Test results *****")
        for index, item in enumerate(predictions):
            item = test_dataset.get_labels()[item]
            writer.write("%s\n" % (item))


if __name__ == "__main__":
    main()
