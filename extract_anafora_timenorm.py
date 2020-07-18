import os
import argparse

import anafora


operator_types = ["This", "Last", "Next", "Before", "After", "Between",
                  "Sum", "Difference", "Union", "Intersection", "Every-Nth",
                  "NthFromStart", "NthFromEnd", "Frequency"]


def remove_item(element, item_name):
    item = element.xml.find(item_name)
    element.xml.remove(item)
    

def convert_xml(xml_path, output_path, raw_path=None):
    data = anafora.AnaforaData.from_file(xml_path)
    wrong_patterns = False

    raw = None
    if raw_path is not None:
        with open(raw_path) as raw_file:
            raw = raw_file.read()
    
    # Remove Events
    annotations_to_delete = list(data.annotations.select_type("Event"))
    for annotation in annotations_to_delete:
        data.annotations.remove(annotation)
    # Remove PreAnnotations
    annotations_to_delete = list(data.annotations.select_type("PreAnnotation"))
    for annotation in annotations_to_delete:
        data.annotations.remove(annotation)
    # Remove NotNormalizable
    annotations_to_delete = list(data.annotations.select_type("NotNormalizable"))
    for annotation in annotations_to_delete:
        data.annotations.remove(annotation)

    # Remove everything but Type
    annotations_by_span = {}
    duplicate_annotations = set()
    for annotation in iter(data.annotations):
        remove_item(annotation, "parentsType")
        remove_item(annotation, "properties")
        for span in annotation.spans:
            if span not in annotations_by_span:
                annotations_by_span[span] = set()
            annotation_types_in_span = [annotation_in_span.type for annotation_in_span in annotations_by_span[span]]
            if annotation.type not in annotation_types_in_span:
                annotations_by_span[span].add(annotation)
            else:
                duplicate_annotations.add(annotation)

    # Remove duplicate annotations
    for annotation in duplicate_annotations:
        data.annotations.remove(annotation)
    
    # Remove implicit operators and unwanted entities
    spans_with_multiple = [span for span in annotations_by_span if len(annotations_by_span[span]) > 1]
    for span in spans_with_multiple:
        annotations = annotations_by_span[span]
        operators = [annotation for annotation in annotations if annotation.type in operator_types]
        num_operators = len(operators)
        num_non_operators = len(annotations) - num_operators

        to_remove = set()
        if num_non_operators > 1:
            periods = [annotation for annotation in annotations if annotation.type == "Period"]
            if len(periods) > 0:
                numbers = [annotation for annotation in annotations if annotation.type == "Number"]
                to_remove.update(numbers)
        if num_operators > 1:
            intersections = [annotation for annotation in annotations if annotation.type == "Intersection"]
            to_remove.update(intersections)            
        if num_operators > 0 and num_non_operators > 0:
            to_remove.update(operators)

        if len(annotations) - len(to_remove) == 1:
            for annotation in to_remove:
                data.annotations.remove(annotation)
        else:
            wrong_patterns = True
            print("Wrong annotation pattern: %s" % annotations)
            if raw is not None:
                start, end = span
                print("TEXT: %s" % raw[start-10:end+10])

    if wrong_patterns:
        print("Data with wrong patterns. File %s will not be saved.\n" % output_path)
    else:
        output_document = os.path.split(output_path)[0]
        if not os.path.exists(output_document):
            os.mkdir(output_document)
        data.to_file(output_path)
        
    
def convert_dir(input_dir, output_dir, raw_dir=None):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for document in anafora.walk(input_dir):
        document_dir = document[0]
        document_name = document[1]

        for xml_name in document[2]:
            if xml_name.endswith(".TimeNorm.gold.completed.xml"):
                xml_path = os.path.join(input_dir, document_dir, xml_name)
                output_path = os.path.join(output_dir, document_name, xml_name)
                raw_path = None
                if raw_dir is not None:
                    raw_path = os.path.join(raw_dir, document_dir, document_name)
                convert_xml(xml_path, output_path, raw_path)

            
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="""%(prog)s converts one directory of Anafora XML annotations 
                                                    into the SemEval-2021 shared task format.""")

    parser.add_argument("-i", "--input", metavar="DIR", dest="input_dir", required=True,
                        help="The root of a set of Anafora XML directories representing reference annotations.")
    parser.add_argument("-o", "--output", metavar="DIR", dest="output_dir", required=True,
                        help="The root of the directory structure where the converted Anafora XML will be stored.")
    parser.add_argument("-r", "--raw", metavar="DIR", dest="raw_dir", required=False,
                        help="The root of directories containing the raw texts for debugging.")
    
    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    raw_dir = args.raw_dir
    convert_dir(input_dir, output_dir, raw_dir=raw_dir)
