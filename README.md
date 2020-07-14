# source-free-domain-adaptation
Scripts to manipulate data sources for the upcoming shared task on clinical negation detection and time expression parsing


## Extracting the i2b2 2010 data for development
The i2b2 2010 Challenge annotated clinical concept spans, with concept type, assertion status, and relations between concepts.
The assertion status could have multiple values: Present, Absent, Hypothetical, Generic, etc.
This task is only concerned with whether a concept is negated or not, so we use the assertion status values but map them in the following way:

Absent -> Negated
Else -> Not negated

Our scripts will create a training input file, a test input file, and a test label fileusing the train and test data from the i2b2 shared task. The train/test input files simply have one instance per line, with special tags around the concept to classify. The test label file has one label per line, -1 for not negated and 1 for negated. The idea of this development data is to use the unlabeled training data to adapt the models we've provided, and then run on the test input file, and generate your own label file with one label per line. You can then check how well you did by giving the test label file and your output label file to the evaluation script.

Steps to get started:

Download and extract three .tar.gz files:
* concept_assertion_relation_training_data.tar.gz
* test_data.tar.gz
* reference_standard_for_test_data.tar.gz

Create a new directory to write the files that you will work with:

``` mkdir <sfda data dir>```

Run the extraction script to populate that directory from the i2b2 2010 input:
    
``` python extract_i2b2_negation.py <i2b2 2010 directory <sfda data dir> ```
    
This will create 3 files in the sfda data directory:
* train.tsv - A file with one instance per line, where an instance is a sentence of context with an entity to train labeled bracketed with xml-like tags: <e> and </e>. The purpose of this file is to use it as unlabeled instances to improve your system.
* dev.tsv - A file with one instance per line, where an instance is as above. The purpose of this file is to test a system that has been adapted on the train.tsv file. The output of your system should look like the dev_labels.txt file described next.
* dev_labels.txt - A file with one label per line, where the lines correspond to the dev.tsv instances. The label 1 corresponds to "negated" and -1 to "not negated" ("Absent" vs. any other assertion label, respectively, in the i2b2 2010 convention). The purpose of this file is first, to show the format required by your system. The second purpose is as the gold input to the scoring script.

