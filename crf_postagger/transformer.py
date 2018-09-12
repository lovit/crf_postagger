bos = 'BOS'
eos = 'EOS'

class AbstractFeatureTransformer:

    def __call__(self, sentence):
        return self.sentence_to_xy(sentence)

    def sentence_to_xy(self, sentence):
        raise NotImplemented

    def potential_function(self, sentence):
        raise NotImplemented

    def to_feature(self, sentence):
        raise NotImplemented

class BaseFeatureTransformer(AbstractFeatureTransformer):
    def __init__(self):
        super().__init__()

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