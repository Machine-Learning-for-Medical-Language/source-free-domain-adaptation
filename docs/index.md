---
layout: sfda
---
## Source-Free Domain Adaptation for Semantic Processing

Data sharing restrictions are common in NLP datasets.
For example, Twitter policies do not allow sharing of tweet text, though tweet IDs may be shared.
The situtation is even more common in clinical NLP, where patient health information must be protected, and annotations over health text, when released at all, often require the signing of complex data use agreements.
The SemEval-2021 Task 10 framework asks participants to develop semantic annotation systems in the face of data sharing constraints.
A participant's goal is to develop an accurate system for a target domain when annotations exist for a related domain but cannot be distributed.
Instead of annotated training data, participants are given a model trained on the annotations.
Then, given unlabeled target domain data, they are asked to make predictions.

We apply this framework to two tasks: **negation detection** and **time expression recognition**.

### Participate

Please join our [Google Group]({{ site.group_url }}) to ask questions and get the most up-to-date information on the task.

Details of the competition, including data sets, models, evaluation, important dates, and how to participate will soon appear on the [SemEval-2021 Task 10 CodaLab site]({{ site.codalab_url }}). We will post to the the Google Group when this is available.

### Organizers
Egoitz Laparra, Yiyun Zhao, [Steven Bethard](https://bethard.faculty.arizona.edu/) (University of Arizona)

[Tim Miller](https://scholar.harvard.edu/tim-miller/home) (Boston Children's Hospital and Harvard Medical School)

[Özlem Uzuner](https://volgenau.gmu.edu/profile/view/444476) (George Mason University)
