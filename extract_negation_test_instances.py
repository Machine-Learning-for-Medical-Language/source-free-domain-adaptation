''' This script is intended for use by _task participants_ to create the test set that they can run their systems on. Run this script from the command line, giving it:
  1) the path to the NOTEEVENTS.csv file from the MIMIC 3 (v 1.4) distribution, which you must download yourself
  2) the path to the annotations span file which the organizers release at the SemEval test set deadline.
This script reads the spans from the test set file (2nd argument) and finds the text in the NOTEEVENTS.csv file (1st argument) to create test instances in the same order in which they appear in the test set file. Full test instances are written to stdout in a BERT-style tsv format.
'''

import os, sys
import csv
import re

def main(args):
    if len(args) < 2:
        sys.stderr.write('Required argument(s): <NOTEEVENTS.csv file> <span file>\n')
        sys.exit(-1)
    
    rows = set()
    spans = []
    with open(args[1], 'rt') as span_file:
        for line in span_file:
            line = line.rstrip()
            spans.append( [int(x) for x in line.split('\t')] )
            rows.add(spans[-1][0])
    
    # we probably can't store all of mimic in memory but we also can't skip around a lot in that huge file so we just have to read through the whole CSV once to get the notes we need.
    rows_to_text = {}
    with open(args[0], 'rt') as notes_file:
        csvreader = csv.reader(notes_file)
        for row_num,row in enumerate(csvreader):
            if row_num == 0:
                continue

            row_id = int(row[0])

            if row_id in rows:
                row_text = row[10]
                rows_to_text[row_id] = row_text
    
    for span in spans:
        row_id, ent_begin, ent_end = span
        text = rows_to_text[row_id]
        inst_begin = max(0, ent_begin-100)
        inst_end = min(ent_end+100, len(text))

        instance_text = text[inst_begin:ent_begin].replace('\n', ' ') + "<e> " + text[ent_begin:ent_end].replace('\n', ' ') + " </e>" + text[ent_end:inst_end].replace('\n', ' ')

        instance_text = re.sub(r'^\S*\s*(.*?)\s*\S*$', r'\1', instance_text)
        print(instance_text)



if __name__ == '__main__':
    main(sys.argv[1:])
