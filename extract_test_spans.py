''' This script is used by the _task organizers_ to extract data from the annotated anafora files and create a span test set, where each row represents one instance, containing the row id (index into MIMIC) for the file and the character offsets.'''

import sys
from anafora import walk, AnaforaData
from os.path import join

def main(args):
    if len(args) < 1:
        sys.stderr.write('Required argument(s): <anafora directory>\n')
        sys.exit(-1)

    for sub_dir, text_name, xml_names in walk(args[0], xml_name_regex='[.]dave\.completed\.xml$'):
        # print("text_name = %s" % text_name)
        assert len(xml_names) == 1
        xml_file = join(args[0], sub_dir, xml_names[0])
        # print("xml file = %s" % (xml_file,))
        anafora_data = AnaforaData.from_file(xml_file)
        for event in anafora_data.annotations.select_type('CuiEntity'):
            negated = event.properties['negated']
            span = event.spans
            assert len(span) == 1
            span = span[0]
            assert len(span) == 2

            # print("Row id %s event with span (%d, %d) is negated=%s" % (text_name, span[0], span[1], str(negated)))
            print('%s\t%d\t%d\t%s' % (text_name, span[0], span[1], str(negated)))

if __name__ == '__main__':
    main(sys.argv[1:])
