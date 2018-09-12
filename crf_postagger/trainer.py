try:
    import pycrfsuite
except:
    print('Failed to import python-crfsuite')

from collections import namedtuple
from .utils import get_process_memory

Feature = namedtuple('Feature', 'idx count')

class Trainer:
    def __init__(self, potential_function, min_count=10,
        l2_cost=1.0, l1_cost=1.0, scan_batch_size=200000,
        max_iter=300, verbose=True):

        self.potential_function = potential_function
        self.min_count = min_count
        self.l2_cost = l2_cost
        self.l1_cost = l1_cost
        self.scan_batch_size = scan_batch_size
        self.max_iter = max_iter
        self.verbose = verbose

    def scan_features(self, sentences, potential_function,
        min_count=2, scan_batch_size=1000000):

        def trim(counter, min_count):
            counter = {
                feature:count for feature, count in counter.items()
                # memorize all words no matter how the word occured.
                if (count >= min_count) or (feature[0] == 'x[0]')
            }
            return counter

        def print_status(i, counter):
            info = '{} sent, {} features, mem={:f} Gb'.format(
                i, len(counter), get_process_memory())
            print('\r[CRF tagger] scanning from {}'.format(info), end='')

        counter = {}

        for i, sentence in enumerate(sentences):
            # remove infrequent features
            if (i % scan_batch_size == 0):
                counter = trim(counter, min_count)
            # transform sentence to features
            sentence_ = potential_function(sentence)
            # count
            for features in sentence_:
                for feature in features:
                    counter[feature] = counter.get(feature, 0) + 1
            # print status
            if self.verbose and i % 10000 == 0:
                print_status(i, counter)

        # last removal of infrequent features
        counter = trim(counter, min_count)

        if self.verbose:
            print_status(i, counter)
            print(' done.')

        return counter

    def train(self, sentences):

        features = self.scan_features(
            sentences, self.potential_function,
            self.min_count, self.scan_batch_size)

        # feature encoder
        self._features = {
            # wrapping feature idx and its count
            feature:Feature(idx, count) for idx, (feature, count) in
            # sort features by their count in decreasing order
            enumerate(sorted(features.items(), key=lambda x:-x[1]
            ))
        }

        # feature id decoder
        self._idx2feature = [
            feature for feature in sorted(
                self._features, key=lambda x:self._features[x].idx)
        ]

        self._train_pycrfsuite(sentences)

    def _train_pycrfsuite(self, sentences):

        def print_status(i):
            info = 'from {} sents, mem = {:f} Gb'.format(
                i, get_process_memory())
            print('\r[CRF tagger] appending features {}'.format(info), end='')

        trainer = pycrfsuite.Trainer(verbose=self.verbose)

        for i, sentence in enumerate(sentences):

            if self.verbose and i % 2000 == 0:
                print_status(i)

            # transform sentence to features
            x = self.potential_function(sentence)
            y = [tag for _, tag in sentence]

            # use only conformed feature
            x = [[xij for xij in xi if xij in self._features] for xi in x]

            trainer.append(x, y)

        if self.verbose:
            print_status(i)
            print(' done.\n[CRF tagger] begin training')

        # set pycrfsuite parameters
        params = {
            'feature.minfreq':max(0,self.min_count),
            'max_iterations':max(1, self.max_iter),
            'c1':max(0, self.l1_cost),
            'c2':max(0, self.l2_cost)
        }
        model_path = '_trained_crf'

        # do train
        trainer.set_params(params)
        trainer.train(model_path)