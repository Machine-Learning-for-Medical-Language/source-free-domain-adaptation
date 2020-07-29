# Time expression recognition baseline

This is  the baseline for the **time expression recognition** task of [SemEval-2020 Task 10: Source-Free Domain Adaptation for Semantic Processing](https://machine-learning-for-medical-language.github.io/source-free-domain-adaptation/). This software allows the participants to download the pre-trained model supplied by the organizers, and make predictions in the required Anafora `.xml`.

The model provided is a sequence tagger, fine-tuned on 25,000+ time expressions in de-identified clinical notes. The following table shows the _in-domain_ and _out-of-domain_ performances, the latter (`practice_data`, `evaluation_data`) are used as the baseline values for the different evaluation phases.

|                 | P     | R     | F1    |
|-----------------|-------|-------|-------|
| in-domain_data  | 0.969 | 0.966 | 0.968 |
| practice_data   | 0.777 | 0.685 | 0.728 |
| evaluation_data |       |       |       |


## Get and prepare data

### Practice Data

The trial data for the practice phase consists of 14 articles from the _AQUAINT_ and _TimeBank_ subsets of  _TempEval-2013_, i.e. _"Newswire"_ domain. For each documents, there is a file ending in _"TimeNorm.gold.completed.xml"_ that contains the annotated temporal expressions following the Anafora schema.

You can download the annotations for this phase [**here**](https://github.com/Machine-Learning-for-Medical-Language/source-free-domain-adaptation/tree/master/practice_data/time).

To get the plain text corresponding to the annotations you need to install [**anaforatools**](https://pypi.org/project/anaforatools/).

The following command will copy the plain text file in each document directory:

    python -m anafora.copy_text --format=timeml /path/to/TBAQ-cleaned/ /path/to/anafora-annotation/TempEval-2013/

Where:

-   _TempEval-2013_ is the directory with the annotated documents
-   _TBAQ-cleaned_ is the directory you get when you unzip [http://www.cs.york.ac.uk/semeval-2013/task1/data/uploads/datasets/tbaq-2013-03.zip](http://www.cs.york.ac.uk/semeval-2013/task1/data/uploads/datasets/tbaq-2013-03.zip)

To make a submission for this phase, you must include the output for the 14 documents following the structure explained in the **Data and Evaluation** section.

## Usage


The first time you run the baseline, the pre-trained model, `clulab/roberta-timex-semeval`, will be automatically downloaded in your computer. If you want to produce some predictions with this model, you need to pass as arguments the directory containing the input text and the target directory where the predictions will be stored. For example, to process the _AQUAINT_ subset from _TempEval-2013_, just run:

    python run_time.py -p /path/to/anafora-annotation/TempEval-2013/AQUAINT -o /path/to/output/AQUAINT [--no_cuda]

Recall that the `anafora-annotation` folder includes both raw text and Anafora annotation files, but it could contain only the former since the latter are not needed to make predictions. This will be the case during the evaluation phase. Use the `--no_cuda` option if you are going to run the model in the gpu.

Run `python run_time.py -h` to explore additional options and arguments you can play with, like the hyperpameters of the model. 

#### Other commands

You can train a model with the following command:

    python run_time.py -t /path/to/train-data/ -s /path/to/save-model/ [--no_cuda]

The `train-data` directory must follow a similar structure to the _AQUAINT_ or _TimeBank_ folders and include, for each document, both the raw text file (with no extension) and the Anafora annotation file (with `.xml` extension). After running the training, the `save-model` will contain two sub-folders, `logs`, with a set of log files that can be visualized with _TensorBoard_, and `results`, that contains all the checkpoints saved during the training and three files (`pytorch_model.bin`, `training_args.bin` and `config.json`) with the configuration and weights of the final model.

By default, this will finetune `clulab/roberta-timex-semeval` model. You can replace it with a model stored in the HuggingFace hub using the `-m` option.E.g. `-m clulab/roberta-base`.

To use this new version of the model for predictions, you can run:

    python run_time.py -p /path/to/text/TempEval-2013/AQUAINT -o /path/to/output/AQUAINT -m /path/to/save-model/results/ [--no_cuda]
