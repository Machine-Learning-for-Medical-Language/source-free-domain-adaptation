import difflib
import glob
import sys
import os
import fnmatch
import collections
from sklearn.metrics import f1_score,precision_score,recall_score
from anafora import evaluate, timeml


def path_lines(root, subdir, pattern, replace=None):
    result = []
    prefix = os.path.join(root, subdir)
    for name in glob.glob(os.path.join(prefix, pattern), recursive=True):
        if os.path.isfile(name):
            name = name[len(prefix) + 1:]
            if replace is not None:
                for old, new in replace.items():
                    name = name.replace(old, new)
            result.append(name + '\n')
    return result


def score_time(ref_domain, res_domain, results):
    scores_type=evaluate.Scores
    exclude=("Event")
    file_named_scores = evaluate.score_dirs(
        reference_dir=ref_domain,
        predicted_dir=res_domain,
        exclude=exclude) # pairwise=True

    all_named_scores = collections.defaultdict(lambda: scores_type())
    for _, named_scores in file_named_scores:
        for name, scores in named_scores.items():
            all_named_scores[name].update(scores)

    results['time_f1']= all_named_scores["*"].f1()
    results['time_prec'] = all_named_scores["*"].precision()
    results['time_recall'] = all_named_scores["*"].recall()

def score_negation(ref_domain,res_domain,results):
    ref = read_tsv(ref_domain)
    res = read_tsv(res_domain)
    assert len(ref) == len(res)
    results['neg_f1']=  f1_score(ref,res,average='micro')
    results['neg_prec'] = precision_score(ref,res,average='micro')
    results['neg_recall'] = recall_score(ref,res,average='micro')

def init_metrics():
    """ initiliazes a dictionary with all the metrics """
    metrics={}
    metrics['neg_prec']=-999.999
    metrics['neg_recall']=-999.999
    metrics['neg_f1']=-999.999
    metrics['time_prec']=-999.999
    metrics['time_recall']=-999.999
    metrics['time_f1']=-999.999
    return metrics

def write_metrics(metrics,output_file):
    """ writes output for Codalab """
    for keys in metrics.iterkeys():
        output_file.write(keys+":{0}\n".format(metrics[keys]))

def read_tsv(file):
    output = []
    with open(file, 'r') as f_output:
        for record in f_output:
            output.append(int(record))
    return output


if __name__ == "__main__":
    _, input_dir, output_dir = sys.argv

    # check which tasks have been submitted
    has_time = os.path.exists(os.path.join(input_dir, 'res', 'time'))
    has_negation = os.path.exists(os.path.join(input_dir, 'res', 'negation'))
    to_system = {'gold': 'system'}

    # exit with an error if any of the expected files were not submitted
    if has_time == has_negation:  # has both or has neither
        expected = path_lines(input_dir, 'ref', "**", to_system)
        uploaded = path_lines(input_dir, 'res', "**")
    elif has_time:
        expected = path_lines(input_dir, 'ref', "time/**", to_system)
        uploaded = path_lines(input_dir, 'res', "time/**")
    else:  # has_negation
        expected = path_lines(input_dir, 'ref', "negation/**", to_system)
        uploaded = path_lines(input_dir, 'res', "negation/**")
    diff = list(difflib.unified_diff(a=expected, b=uploaded, n=0,
                                     fromfile="expected", tofile="uploaded"))
    if diff:
        sys.stderr.write("Incorrect files:\n")
        sys.stderr.writelines(diff)
        sys.exit(1)

    # scoring
    metrics = init_metrics()
    if has_time:
        ref_domain = os.path.join(input_dir, 'ref', 'time')
        res_domain = os.path.join(input_dir, 'res', 'time')
        score_time(ref_domain, res_domain, metrics)
    if has_negation:
        ref_domain = os.path.join(input_dir, 'ref', 'negation/gold.tsv')
        res_domain = os.path.join(input_dir, 'res', 'negation/system.tsv')
        score_negation(ref_domain, res_domain, metrics)

    # write scores file
    with open(os.path.join(output_dir, "scores.txt"), "w") as output_file:
        write_metrics(metrics, output_file)
