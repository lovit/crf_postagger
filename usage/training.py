import sys
sys.path.append('../')

import crf_postagger
from crf_postagger import Corpus
from crf_postagger import BaseFeatureTransformer

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus_path', type=str, default='./', help='corpus path')
    parser.add_argument('--corpus_length', type=int, default=-1, help='if you set, sample sentence')
    parser.add_argument('--max_iter', type=int, default=100, help='the number of maximal iteration of CRF')
    parser.add_argument('--model_path', type=str, default='./crf_tagger.json', help='trained model path')
    parser.add_argument('--verbose', dest='verbose', action='store_true')

    args = parser.parse_args()
    corpus_path = args.corpus_path
    corpus_length = args.corpus_length
    max_iter = args.max_iter
    model_path = args.model_path
    verbose = args.verbose

    trainer = Trainer(
        Corpus(corpus_path, num_sent=corpus_length),
        max_iter = max_iter,
        l1_cost = 0,
        verbose = verbosd
    )
    trainer._save_as_json(model_path)

if __name__ == '__main__':
    main()