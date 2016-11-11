#!/usr/bin/env python

from nalaf.utils.readers import HTMLReader
from nalaf.utils.annotation_readers import AnnJsonAnnotationReader
from nalaf.structures.relation_pipelines import RelationExtractionPipeline
from nalaf.preprocessing.tokenizers import TmVarTokenizer, NLTK_TOKENIZER
from nalaf.preprocessing.spliters import NLTKSplitter
from nalaf.preprocessing.parsers import SpacyParser
from spacy.en import English

import argparse

"""
Extracts the number of true relations present in a document for following cases
 D0 - relations in Same sentences.
 D1 - relations in sentences which are 1 sentence apart.
 D2 - relations in sentences which are 2 sentence apart.
 D3 - relations in sentences which are 3 sentence apart.
 D4+ - relations in sentences which are 4 or more than 4 sentence apart.
"""

# Parse command line arguments
def parse_arguments(argv):

    #TODO: Parse the argument.
    parser = argparse.ArgumentParser(description='Simple-evaluate relna corpus corpus')

    # parser.add_argument('--D0', default="yes", choices=["yes", "no"])
    # parser.add_argument('--D1', default="yes", choices=["yes", "no"])
    # parser.add_argument('--D2', default="yes", choices=["yes", "no"])
    # parser.add_argument('--D3', default="yes", choices=["yes", "no"])
    # TODO : Add necessary command line arguments
    parser.add_argument('--corpus', default="relna", choices=["LocText", "relna"])
    parser.add_argument('--use_tk', default=False, action='store_true')
    parser.add_argument('--use_test_set', default=False, action='store_true')
    parser.add_argument('--use_full_corpus', default=True, action='store_true')

    args = parser.parse_args(argv)

    print(args)

    return args

# Extract all true relations
def RelationExtractor(argv=None):
    argv = [] if argv is None else argv
    args = parse_arguments(argv)

    if args.use_tk:
        # svm_folder = ''  # '/usr/local/manual/svm-light-TK-1.2.1/' -- must be in your path
        nlp = English(entity=False)
        parser = SpacyParser(nlp, constituency_parser=True)
    else:
        # svm_folder = ''  # '/usr/local/manual/bin/' -- must be in your path
        parser = None

    if args.corpus == "relna":
        # Relna
        dataset_folder_html = '../../resources/corpora/relna/corrected/'
        dataset_folder_annjson = dataset_folder_html
        rel_type = 'r_4'

    elif args.corpus == "LocText":
        # LocText
        dataset_folder_html = '../../resources/corpora/LocText/LocText_plain_html/pool/'
        dataset_folder_annjson = '../../resources/corpora/LocText/LocText_master_json/pool/'
        rel_type = 'r_5'

    def read_dataset():
        dataset = HTMLReader(dataset_folder_html).read()
        AnnJsonAnnotationReader(
                dataset_folder_annjson,
                read_only_class_id=None,
                read_relations=True,
                delete_incomplete_docs=False).annotate(dataset)

        return dataset

    # Extracts true relations in D0, D1, D2 and D3
    def Extract_True_Relation_For_D0_D1_D2_D3():

        if args.use_full_corpus:
            dataset = read_dataset()
        else:
            dataset, _ = read_dataset().percentage_split(0.1)

        extractor = RelationExtractionPipeline('e_1', 'e_2', rel_type, parser=parser, splitter=NLTKSplitter(), tokenizer=NLTK_TOKENIZER)
        #
        # try:
        #     gen = dataset.tokens()
        #     next(gen)
        # except StopIteration:
        #     pipeline.splitter.split(dataset)
        #     pipeline.tokenizer.tokenize(dataset)

        # D0 - relations in Same sentences.
        # count_of_D0 = pipeline.edge_generator.count_relations_in_D0(dataset)

        # D1 - relations in sentences which are 1 sentence apart.
        # count_of_D1, count_of_D2, count_of_D3, count_of_D4_plus = pipeline.edge_generator.count_relations_in_D1_D2_D3_D4(dataset)

        # all_sentences = list(dataset.sentences())
        # Relation (class) is unhashable.
        # set_of_relations = set(all_relations)
        # print(set_of_relations)
        # print(all_relations)

        # all_annotations = list(dataset.all_annotations_with_ids())
        # print(all_annotations)

        extractor.splitter.split(dataset)
        extractor.tokenizer.tokenize(dataset)
        all_relations = list(dataset.relations())
        all_parts = list(dataset.parts())

        count_of_D0 = 0
        count_of_D1 = 0
        count_of_D2 = 0
        count_of_D3 = 0
        count_of_D4_plus = 0

        for relation in all_relations:
            for part in all_parts:
                e1_index = part.get_sentence_index_for_annotation(relation.entity1[0])
                e2_index = part.get_sentence_index_for_annotation(relation.entity2[0])

                if abs(e1_index - e2_index) == 0:
                    count_of_D0+=1
                elif abs(e1_index - e2_index) == 1:
                    count_of_D1+=1
                elif abs(e1_index - e2_index) == 2:
                    count_of_D2 += 1
                elif abs(e1_index - e2_index) == 3:
                    count_of_D3 += 1
                else:
                    count_of_D4_plus += 1

        # for part in dataset.parts():
        #     for rel in part.relations:
        #         print(rel.en)
        #         print(part.get_sentence_index_for_annotation(rel.entity1[0]))
        #         print(part.get_sentence_index_for_annotation(rel.entity2[0]))

                # print(rel.entity1)
                # print(rel.entity2)
        # print("All relations")
        # print(len(all_relations))

        # print("All Sentences \n")
        # print(len(all_sentences))

        print("\n ********************* True relation count is as follows **********************\n")
        print("\n D0 - relations in Same sentences -----------------------------------------> ", count_of_D0)
        print("\n D1 - relations in sentences which are 1 sentence apart. ------------------> ", count_of_D1)
        print("\n D2 - relations in sentences which are 2 sentence apart. ------------------> ", count_of_D2)
        print("\n D3 - relations in sentences which are 3 sentence apart. ------------------> ", count_of_D3)
        print("\n D4+ - relations in sentences which are 4 or more than 4 sentence apart ---> ", count_of_D4_plus)
        print("\n ************************* END *************************\n")

    # A call to extract true relations
    Extract_True_Relation_For_D0_D1_D2_D3()

if __name__ == "__main__":
    import sys
    RelationExtractor(sys.argv[1:])

