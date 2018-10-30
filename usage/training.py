import argparse
import sys
sys.path.append('../')

import crf_postagger
from crf_postagger import Corpus
from crf_postagger import Trainer
from crf_postagger import BaseFeatureTransformer
from crf_postagger.trigram import TrigramFeatureTransformer
from crf_postagger.hmm_style import HMMStyleFeatureTransformer

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus_path', type=str, default='../data/sejong_corpus_lr_sepxsv.txt', help='corpus path')
    parser.add_argument('--corpus_length', type=int, default=-1, help='if you set, sample sentence')
    parser.add_argument('--feature_type', type=str, default='trigram',
                        choices=['base', 'hmm_style', 'trigram']
                       )
    parser.add_argument('--max_iter', type=int, default=100, help='the number of maximal iteration of CRF')
    parser.add_argument('--model_path', type=str, default='../trigram_sejong_lr_sepxsv.json', help='trained model path')
    parser.add_argument('--verbose', dest='verbose', action='store_true')

    args = parser.parse_args()
    corpus_path = args.corpus_path
    corpus_length = args.corpus_length
    feature_type = args.feature_type
    max_iter = args.max_iter
    model_path = args.model_path
    verbose = args.verbose

    if feature_type == 'hmm_style':
        sentence_to_xy = HMMStyleFeatureTransformer()
    elif feature_type == 'trigram':
        sentence_to_xy = TrigramFeatureTransformer()
    else:
        sentence_to_xy = BaseFeatureTransformer()

    trainer = Trainer(
        Corpus(corpus_path, num_sent=corpus_length),
        sentence_to_xy = sentence_to_xy,
        max_iter = max_iter,
        l1_cost = 0,
        verbose = verbose
    )
    trainer._save_as_json(model_path)

if __name__ == '__main__':
    main()