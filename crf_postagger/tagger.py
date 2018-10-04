from .utils import bos, eos, unk, Eojeols
from .transformer import *

class AbstractTagger:

    def __init__(self, parameters,
        feature_transformer=None, verbose=False):

        if verbose:
            print('use {}'.format(feature_transformer.__class__))

        self.parameters = parameters
        self.feature_transformer = feature_transformer
        self.verbose = verbose

    def evaluate(self, wordpos_sentence, debug=False):

        # feature transform
        sentence_, tags = self.feature_transformer(wordpos_sentence)
        score = 0

        # transition weight
        for s0, s1 in zip(tags, tags[1:]):
            transition = (s0, s1)
            coef = self.parameters.transitions.get(transition, 0)
            if debug:
                print('{} = {:f}, score = {:f}'.format(transition, coef, score))
            score += coef

        # state feature weight
        for features, tag in zip(sentence_, tags):
            for feature in features:
                if debug:
                    print('{} -> {} = {:f}, score = {:f}'.format(
                        feature, tag, coef, score))
                coef = self.parameters.state_features.get((feature, tag), 0)
                score += coef

        return score

    def tag(self, sentence, flatten=True, debug=False):
        raise NotImplemented

    def add_user_dictionary(self, tag, word_score):
        if not (tag in self.parameters.pos2words):
            raise ValueError('{} tag does not exist in model'.format(tag))
        for word, score in word_score.items():
            self.parameters.pos2words[tag][word] = score

    def _remain_details(self, eojeols):
        return [(eojeol.pos, eojeol.begin, eojeol.end, eojeol.eojeol_score)
                for eojeol in eojeols.eojeols[1:-1]]

    def _remain_only_pos(self, eojeols):
        poses = []
        for eojeol in eojeols.eojeols[1:-1]:
            poses.append((eojeol.first_word, eojeol.first_tag))
            if eojeol.is_compound:
                poses.append((eojeol.last_word, eojeol.last_tag))
        return poses