import os
import argparse
import torch
import numpy as np
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import precision_recall_fscore_support

from utils_time import read_labels, init_nlp_pipeline, create_datasets, write_predictions


class Model:

    def __init__(self, args):
        self.bio_mode = not args.io_mode
        self.labels = read_labels("resources/labels.txt", bio_mode=self.bio_mode)
        config = AutoConfig.from_pretrained(args.model_name_or_path)
        config.num_labels = len(self.labels)
        config.id2label = dict((idx, label) for idx, label in enumerate(self.labels))
        config.label2id = dict((label, idx) for idx, label in enumerate(self.labels))
        self.label_pad_id = torch.nn.CrossEntropyLoss().ignore_index
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base", config=config,
                                                       vocab_file="resources/roberta-base-vocab-modified.json",
                                                       merges_file="resources/roberta-base-merges-modified.txt",
                                                       use_fast=True)
        self.model = AutoModelForTokenClassification.from_pretrained("roberta-base", config=config)
        results_path = os.path.join(args.save_dir, "results")
        logs_path = os.path.join(args.save_dir, "logs")
        self.args = TrainingArguments(
            output_dir=results_path,
            logging_dir=logs_path,
            no_cuda=args.no_cuda,
            seed=args.seed,
            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=args.train_batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps,
            weight_decay=args.weight_decay,
            adam_epsilon=args.adam_epsilon,
            save_steps=args.save_steps,
            evaluate_during_training=args.validate,
            eval_steps=args.eval_steps,
            max_grad_norm=args.max_grad_norm,
            local_rank=args.local_rank,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            fp16=args.fp16,
            fp16_opt_level=args.fp16_opt_level
        )

    @staticmethod
    def train_data_collator(features):
        batch = dict()
        batch["input_ids"] = torch.stack([f.input_ids for f in features])
        batch["attention_mask"] = torch.stack([f.attention_mask for f in features])
        batch["labels"] = torch.stack([f.label for f in features])
        return batch

    @staticmethod
    def test_data_collator(features):
        batch = dict()
        batch["input_ids"] = torch.stack([f.input_ids for f in features])
        batch["attention_mask"] = torch.stack([f.attention_mask for f in features])
        return batch

    @staticmethod
    def compute_metrics(prediction):
        labels = prediction.label_ids.flatten()
        predictions = prediction.predictions.argmax(-1).flatten()
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="micro")
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def train(self, train_dataset, eval_dataset=None):
        trainer = Trainer(
            model=self.model,
            args=self.args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics,
            data_collator=self.train_data_collator
        )
        trainer.train()
        trainer.save_model()

    def predict(self, dataset):
        trainer = Trainer(
            model=self.model,
            args=self.args,
            data_collator=self.test_data_collator
        )
        prediction, _, _ = trainer.predict(dataset)
        dataset.prediction = np.argmax(prediction, axis=2)
        if self.bio_mode:
            dataset.prediction += dataset.prediction % 2


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="""%(prog)s runs SEMEVAL-2021 temporal baseline.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-t", "--train", metavar="DIR", dest="train_dir",
                        help="The root of the training set directory tree containing raw text and of Anafora XML.")
    parser.add_argument("-v", "--valid", metavar="DIR", dest="valid_dir",
                        help="The root of the validation set directory tree containing raw text and of Anafora XML.")
    parser.add_argument("-p", "--predict", metavar="DIR", dest="predict_dir",
                        help="The root of the directory tree containing raw text for prediction.")
    parser.add_argument("-o", "--output", metavar="DIR", dest="output_dir",
                        help="The directory to store the prediction in Anafora XML.")
    parser.add_argument("-s", "--save", metavar="DIR", dest="save_dir", default="./",
                        help="The directory to save the model and the log files.")
    parser.add_argument("-m", "--model", metavar="NAME|PATH", dest="model_name_or_path", default="roberta-base",
                        help="Name or path ot a trained model.")
    parser.add_argument("-i", "--io", dest="io_mode", action="store_true",
                        help="Use IO labelling instead of BIO.")

    hyper = parser.add_argument_group("hyper_parameters")
    hyper.add_argument("--no_cuda", action="store_true", help=" ")
    hyper.add_argument("--max_seq_length", default=128, type=int, help=" ")
    hyper.add_argument("--train_batch_size", default=32, type=int, help=" ")
    hyper.add_argument("--eval_batch_size", default=8, type=int, help=" ")
    hyper.add_argument("--learning_rate", default=5e-5, type=float, help=" ")
    hyper.add_argument("--num_train_epochs", default=3.0, type=float, help=" ")
    hyper.add_argument("--warmup_steps", default=500, type=int, help=" ")
    hyper.add_argument("--weight_decay", default=0.01, type=float, help=" ")
    hyper.add_argument("--adam_epsilon", default=1e-8, type=float, help=" ")
    hyper.add_argument("--save_steps", default=500, type=int, help=" ")
    hyper.add_argument("--eval_steps", default=500, type=int, help=" ")
    hyper.add_argument("--validate", action="store_true", help=" ")
    hyper.add_argument("--max_grad_norm", default=1.0, type=float, help=" ")
    hyper.add_argument("--local_rank", default=-1, type=int, help=" ")
    hyper.add_argument("--seed", default=42, type=int, help=" ")
    hyper.add_argument("--gradient_accumulation_steps", default=1, type=int, help=" ")
    hyper.add_argument("--fp16", action="store_true", help=" ")
    hyper.add_argument("--fp16_opt_level", default="O1", type=str, help=" ")

    args = parser.parse_args()
    train_path = args.train_dir
    valid_path = args.valid_dir
    predict_path = args.predict_dir
    output_path = args.output_dir

    model = Model(args)
    nlp = init_nlp_pipeline()

    if train_path is not None and valid_path is not None:
        train_dataset = create_datasets(model, nlp, train_path, train=True)
        valid_dataset = create_datasets(model, nlp, valid_path, train=True)
        model.train(train_dataset, eval_dataset=valid_dataset)
        model.predict(valid_dataset)
        write_predictions(model, valid_dataset, output_path)

    elif train_path is not None:
        train_dataset = create_datasets(model, nlp, train_path, train=True)
        model.train(train_dataset)

    elif predict_path is not None:
        test_dataset = create_datasets(model, nlp, predict_path)
        model.predict(test_dataset)
        write_predictions(model, test_dataset, output_path)
