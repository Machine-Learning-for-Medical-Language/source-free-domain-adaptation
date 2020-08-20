import os, sys
from os.path import join
from os import path, listdir
import re

class Concept:
    def __init__(self, term, start_line, start_token, end_line, end_token, negated):
        self.term = term
        self.start_line = start_line
        self.start_token = start_token
        self.end_line = end_line
        self.end_token = end_token
        self.negated = negated

concept_patt = re.compile('c="(.*)" (\d+):(\d+) (\d+):(\d+)')
test_files = ['0002','0006','0010','0014','0018','0022','0026','0030','0034','0038','0042','0046','0050','0054','0058','0062','0066','0070','0074','0078','0082','0086','0090','0094','0098','0102','0106','0110','0114','0118','0122','0126','0130','0134','0138','0142','0146','0150','0154','0158','0162','0166','0170','0174','0178','0182','0186','0190','0194','0198','0202','0206','0210','0214','0218','0222','0226','0230','0234','0238','0242','0246','0250','0254','0258','0262','0266','0270','0274','0278','0282','0286','0290','0294','0298','0302','0306','0310','0314','0318','0322','0326','0330','0334','0338','0342','0346','0350','0354','0358','0362','0366','0370','0374','0378','0382','0386','0390','0394','0398','0402','0406','0410','0413','0416','0419','0422','0425','0428','0431','0434','0437','0440','0443','0446','0449','0452','0455','0458','0461','0463','0465','0467','0468','0469','0470','0471','0472','0473','0474','0475','0476','0477']

def get_file_concepts(con_fn):
    concepts = []
    with open(con_fn, 'rt') as con_file:
        for line in con_file.readlines():
            concept, con_type, assertion = line.rstrip().split('||')
            if 'absent' in assertion:
                label = 1
            else:
                label = -1

            m = concept_patt.match(concept)
            if m is None:
                sys.stderr.write("This line doesn't match the concept regex!")
                continue

            term = m.group(1)
            start_line = int(m.group(2))
            start_token = int(m.group(3))
            end_line = int(m.group(4))
            end_token = int(m.group(5))

            concept = Concept(term, start_line, start_token, end_line, end_token, label)
            concepts.append(concept)
    return concepts

def get_text_lines(text_fn):
    text_lines = []
    with open(text_fn, 'rt') as txt_file:
        for line in txt_file.readlines():
            tokens = line.rstrip().split()
            text_lines.append(tokens)

    return text_lines

def get_inst_lines(concept, text_lines):
    inst_lines = []
    for line_num in range(concept.start_line, concept.end_line+1):
        tokens = text_lines[line_num-1].copy()

        if line_num == concept.end_line:
            tokens.insert(concept.end_token+1, '</e>')
        if line_num == concept.start_line:
            tokens.insert(concept.start_token, '<e>')

        inst_lines.append(' '.join(tokens))
        
    return inst_lines
    
def main(args):
    if len(args) < 2:
        sys.stderr.write('2 required arguments: <i2b2 data directory> <output directory>\n')
        sys.exit(-1)
        
    train_dir = join(args[0], 'concept_assertion_relation_training_data')
    test_note_dir = join(args[0], 'test_data')
    test_label_dir = join(args[0], 'reference_standard_for_test_data')
    
    if not path.exists(args[1]):
        sys.stderr.write('Output directory %s does not exist -- creating for you.\n' % (args[1]))
        os.makedirs(args[1])
    
    if not path.exists(train_dir) or not path.exists(test_note_dir) or not path.exists(test_label_dir):
        sys.stderr.write('''The input directory does not contain all of the required sub-directories: 
            1) concept_assertion_relation_training_data
            2) reference_standard_for_test_data
            3) test_data\n''')
        sys.exit(-1)
    
    
    with open(join(args[1], 'dev_labels.txt'), 'wt') as outl:
        with open(join(args[1], 'dev.tsv'), 'wt') as outf:
            ## Extracting the unlabeled training set (leaving out beth which may overlap mimic)
            concept_dir = join(test_label_dir, 'ast')
#             text_dir = join(test_note_dir, subdir, 'txt')

            files = listdir(concept_dir)
            files.sort()
            for file in files:
                record_name = file[:-4]
                if not file.endswith('.ast'):
                    continue

                if not record_name in test_files:
                    continue
                
                # Create concept data structure for each concept annotation.
                concepts = get_file_concepts(join(concept_dir, file))

                # Now go through text and find the sentence context for each concept we found:
                text_fn = join(test_note_dir, '%s.txt' % (record_name))
                text_lines = get_text_lines(text_fn)
                
                for concept in concepts:
                    inst_lines = get_inst_lines(concept, text_lines)                    

                    outf.write('<cr>'.join(inst_lines))
                    outf.write('\n')
                    outl.write(str(concept.negated))
                    outl.write('\n')
            
    with open(join(args[1], 'train.tsv'), 'wt') as outf:
        for subdir in ('partners',):
            concept_dir = join(train_dir, subdir, 'ast')
            text_dir = join(train_dir, subdir, 'txt')
           
            files = listdir(concept_dir) 
            files.sort()
            for file in files:
                if not file.endswith('ast'):
                    continue
                    
                concepts = get_file_concepts(join(concept_dir, file))
                
                text_fn = join(text_dir, '%s.txt' % (file[:-4]))
                text_lines = get_text_lines(text_fn)
                
                for concept in concepts:
                    inst_lines = get_inst_lines(concept, text_lines)
                    outf.write('<cr>'.join(inst_lines))
                    outf.write('\n')
                    ## We don't write labels for the train data because we want to emphasize that this is an unsupervised domain adaptation setting
                    
                    
if __name__ == '__main__':
    main(sys.argv[1:])
