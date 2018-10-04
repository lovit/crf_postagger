from collections import defaultdict
from ._beam import beam_search
from ._beam import _preference_penalty
from ._beam import _trigram_score
from .. import AbstractTagger
from .. import AbstractParameter
from .. import AbstractFeatureTransformer


class TrigramTagger(AbstractTagger):

    def __init__(self, parameters,
        feature_transformer=None, verbose=False):

        if feature_transformer is None:
            feature_transformer = TrigramFeatureTransformer()
        if parameters is None:
            parameters = TrigramFeatureTransformer()

        self._a_syllable_penalty = -0.3
        self._noun_preference = 0.5
        self._longer_noun_preference = 0.2

        self._beam_score_functions = [
            _preference_penalty,
            _trigram_score
        ]
        super().__init__(parameters, feature_transformer, verbose)

    def tag(self, sentence, flatten=True, guess_tag=False, beam_size=5):
        # generate nodes and edges
        begin_index = self.parameters.generate(sentence, guess_tag)

        # find optimal path
        chars = sentence.replace(' ', '')
        top_eojeols = beam_search(
            begin_index, beam_size, chars, self.parameters,
            self._beam_score_functions, self.parameters.unknown_penalty,
            a_syllable_penalty = self._a_syllable_penalty,
            noun_preference = self._noun_preference,
            longer_noun_preference = self._longer_noun_preference
        )

        # post-processing
        def postprocessing(eojeols, flatten):
            if flatten:
                return self._remain_only_pos(eojeols)
            else:
                return self._remain_details(eojeols)

        top_poses = [(postprocessing(eojeols, flatten), eojeols.score)
                     for eojeols in top_eojeols]

        return top_poses

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

class TrigramParameter(AbstractParameter):
    def __init__(self, model_path=None, pos2words=None, preanalyzed_eojeols=None,
        max_word_len=0, parameter_marker=' -> ', unknown_penalty=-0.1):

        super().__init__(model_path, pos2words, preanalyzed_eojeols,
            max_word_len, parameter_marker, unknown_penalty)

        self._separate_features()

    def _separate_features(self):
        is_1X0 =    lambda x: ('x[-1:0]' in x) and not (' ' in x)
        is_X0_1Y =  lambda x: ('y[-1]' in x) and not (' ' in x)
        is_X01 =    lambda x: ('x[0:1]') in x and not (' ' in x)
        is_X01_Y1 = lambda x: ('x[0:1]') in x and ('y[1]' in x)
        is_1X1 =    lambda x: ('x[-1,1]' in x)
        is_1X01 =   lambda x: ('x[-1:1]' in x)

        def parse_word(feature):
            poses = feature.split(', ')
            wordtags = tuple(
                wordtag for pos in poses for wordtag
                in pos.split(']=')[-1].split('-')
            )
            return wordtags

        # previous features
        self.previous_1X0 = defaultdict(lambda: {})
        self.previous_X0_1Y = defaultdict(lambda: {})

        # successive features
        self.successive_X01 = defaultdict(lambda: {})
        self.successive_X01_Y1 = defaultdict(lambda: {})

        # bothside_features
        self.bothside_1X1 = defaultdict(lambda: {})
        self.bothside_1X01 = defaultdict(lambda: {})

        for (feature, tag), coef in self.state_features.items():
            if is_1X0(feature):
                self.previous_1X0[tag][parse_word(feature)] = coef
            elif is_X0_1Y(feature):
                self.previous_X0_1Y[tag][parse_word(feature)] = coef
            elif is_X01(feature):
                self.successive_X01[tag][parse_word(feature)] = coef
            elif is_X01_Y1(feature):
                self.successive_X01_Y1[tag][parse_word(feature)] = coef
            elif is_1X1(feature):
                self.bothside_1X1[tag][parse_word(feature)] = coef
            elif is_1X01(feature):
                self.bothside_1X01[tag][parse_word(feature)] = coef

        # previous features
        self.previous_1X0 = dict(self.previous_1X0)
        self.previous_X0_1Y = dict(self.previous_X0_1Y)
        # successive features
        self.successive_X01 = dict(self.successive_X01)
        self.successive_X01_Y1 = dict(self.successive_X01_Y1)
        # bothside_features
        self.bothside_1X1 = dict(self.bothside_1X1)
        self.bothside_1X01 = dict(self.bothside_1X01)

    #def generate(self, sentence):
    #    raise NotImplemented