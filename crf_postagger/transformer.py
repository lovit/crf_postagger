from pprint import pprint
from .common import bos, eos

class AbstractFeatureTransformer:

    def __call__(self, sentence):
        return self.sentence_to_xy(sentence)

    def sentence_to_xy(self, sentence):
        """Feature transformer

        :param list_of_tuple pos: a sentence [(word, tag), (word, tag), ...]
        """

        words, tags = zip(*sentence)
        words_ = tuple((bos, *words, eos))
        tags_ = tuple((bos, *tags, eos))

        encoded_sentence = self.potential_function(words_, tags_)
        return encoded_sentence, tags

    def potential_function(self, words_, tags_):
        n = len(tags_) - 2 # except bos & eos
        sentence_ = [self.to_feature(words_, tags_, i) for i in range(1, n+1)]
        return sentence_

    def to_feature(self, sentence):
        raise NotImplemented

    def show_example(self):
        sentence = [
            ('이것', 'Noun'),
            ('은', 'Josa'),
            ('예문', 'Noun'),
            ('이', 'Adjective'),
            ('ㅂ니다', 'Eomi')
        ]
        words, tags = zip(*sentence)
        features, tags_ = self.sentence_to_xy(sentence)
        print('CRF feature example with 5 words\words = {}\ntags={}\nfeatures'.format(words, tags))
        pprint(features)

class BaseFeatureTransformer(AbstractFeatureTransformer):
    def __init__(self):
        super().__init__()

    def to_feature(self, words_, tags_, i):
        features = [
            'x[0]=%s' % words_[i],
            'x[0]=%s, y[-1]=%s' % (words_[i], tags_[i-1]),
            'x[-1:0]=%s-%s' % (words_[i-1], words_[i]),
            'x[-1:0]=%s-%s, y[-1]=%s' % (words_[i-1], words_[i], tags_[i-1]),
            'x[-1,1]=%s-%s' % (words_[i-1], words_[i+1]),
            'x[-1,1]=%s-%s, y[-1]=%s' % (words_[i-1], words_[i+1], tags_[i-1])
        ]
        return features

class TrigramFeatureTransformer(AbstractFeatureTransformer):
    def __init__(self):
        super().__init__()

    def to_feature(self, words_, tags_, i):
        features = [
            # Capital: successive direction, lower case: previous
            # word feature; X0
            'x[0]=%s' % words_[i],
            # previous features; X0_y1, x10
            'x[0]=%s, y[-1]=%s' % (words_[i], tags_[i-1]),
            'x[-1:0]=%s-%s' % (words_[i-1], words_[i]),
            # successive features; X01, X01_Y1
            'x[0:1]=%s-%s' % (words_[i], words_[i+1]),
            'x[0:1]=%s-%s, y[1]=%s' % (words_[i], words_[i+1], tags_[i+1]),
            # both_side; X11, X101
            'x[-1,1]=%s-%s' % (words_[i-1], words_[i+1]),
            'x[-1:1]=%s-%s-%s' % (words_[i-1], words_[i], words_[i+1])
        ]
        return features

class HMMStyleFeatureTransformer(AbstractFeatureTransformer):
    def __init__(self):
        super().__init__()

    def to_feature(self, words_, tags_, i):
        features = [
            'x[0]=%s' % words_[i],
        ]
        return features