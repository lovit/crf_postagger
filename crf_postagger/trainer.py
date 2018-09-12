try:
    import pycrfsuite
except:
    print('Failed to import python-crfsuite')

from .utils import get_process_memory

class Trainer:
    def __init__(self, potential_function, min_count=10,
        l2_cost=1.0, l1_cost=1.0, scan_batch_size=200000,
        verbose=True):

        self.potential_function = potential_function
        self.min_count = min_count
        self.l2_cost = l2_cost
        self.l1_cost = l1_cost
        self.scan_batch_size = scan_batch_size
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

        features = self.scan_features(sentences)

        raise NotImplemented

def sent_to_xy(sentence, potential_function):
    raise NotImplemented