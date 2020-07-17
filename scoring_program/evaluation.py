import sys
import os
import fnmatch
import collections
from sklearn.metrics import f1_score,precision_score,recall_score
from anafora import evaluate, timeml

def dir_struct(path):
    struct = dict()
    for root, dirs, files in os.walk(path):
        root = root.replace(path,'').split('/')[-1]
        struct[root] = list()
        if len(dirs) > 0:
            struct[root].extend(dirs)
        if len(files) > 0:
            struct[root].extend(files)    
    return struct

def check_task(sub_struct, ref_struct):
    participated_tasks = []
    #path_struct = dir_struct(path)
    
    # check whether the dire contain at least one sub task
    for task in sub_struct['']:
        if task in ref_struct['']:
            participated_tasks.append(task)
    if not participated_tasks:
        sys.exit('Error: not task submission was found in the submission, please include ({})'.format(','.join(ref_struct[''])))
        #if task not in struct['']: sys.exit("Error: Wrong domain directory %s" % domain)

    return participated_tasks

def check_dir(sub_struct,ref_struct,task):
    datasets_ref = ref_struct[task]
    datasets_sub = sub_struct[task]
    
    if task == 'time':
        # check whether the two datasets are there
        if not (set(datasets_sub) == set(datasets_ref)):
            sys.exit('Error: expected two datasets ({}) but found ({})'.format(','.join(datasets_ref), ','.join(datasets_sub)))

        # check whether all the subdirs in each dataset and all the files within each subdir
        for dataset in datasets_ref:
        
            dirs_ref = ref_struct[dataset]
            dirs_sub = sub_struct[dataset]
        
            for folder in dirs_ref:
                if folder not in dirs_sub:
                    sys.exit('Error: the subdirectory ({}) is missing in dataset ({}) in task ({}) in the submission'.format(folder,dataset,task))
                else:
                    expected_sub_file = ref_struct[folder][0].replace('gold','system')
                    if expected_sub_file not in sub_struct[folder]:
                        sys.exit('Error: the file ({}) is missing in the folder ({}) in the dataset({}) in the task ({}) in the submission'.format(expected_sub_file,folder,dataset,task))
        
            # if the above sucessfully implemented it means that all the ref are there
            if len(dirs_sub) != len(dirs_ref):
                unexpected = list(set(dirs_sub) - set(dirs_ref))
                sys.exit('Error: unexpected subdirectory(ies): {} in the dataset({}) in task ({})'.format(','.join(unexpected),dataset,task))
    
    if task == 'negation':
        expected_files = []
        for dataset in datasets_ref:
            expected =  dataset.replace('gold','system')
            expected_files.append(expected)
            if expected not in datasets_sub:
                sys.exit('Error, the file({}) is missing in the task({})'.format(expected,task))

        # if the above sucessfully implemented it means that all the ref are there
            if len(expected_files) != len(datasets_sub):
                unexpected = list(set(expected_files) - set(datasets_sub))
                sys.exit('Error: unexpected subdirectory(ies): {} in task ({})'.format(','.join(unexpected),task))
    
    return True

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

    # set out the path
    ref = os.path.join(input_dir, "ref")
    res = os.path.join(input_dir, "res")
    output = os.path.join(output_dir, "scores.txt")

    # file system and checking
    ref_struct = dir_struct(ref)
    res_struct = dir_struct(res)
    tasks = check_task(res_struct,ref_struct)
    for task in tasks: check_dir(res_struct,ref_struct,task)

    # scoring
    metrics = init_metrics()
    if 'time' in tasks:
        ref_domain = os.path.join(ref,'time')
        res_domain = os.path.join(res,'time')
        score_time(ref_domain, res_domain, metrics)
    if 'negation' in tasks:
        ref_domain = os.path.join(ref,'negation/gold.tsv')
        res_domain = os.path.join(res,'negation/system.tsv')
        score_negation(ref_domain, res_domain, metrics)
    output_file=open(output,"w")
    write_metrics(metrics,output_file)
    output_file.close()
