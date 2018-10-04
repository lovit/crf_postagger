from .beam import beam_search
from .common import bos, eos, unk, Eojeols
from .transformer import *
from .path import ford_list
from .utils import _to_end_index
from ._hmm_style import _hmm_style_tagger_weight

class HMMStyleTagger:

    def __init__(self, parameters,
        feature_transformer=None, verbose=False):

        if feature_transformer is None:
            feature_transformer = BaseFeatureTransformer()
        if parameters is None:
            parameters = HMMStyleParameter()
        if verbose:
            print('use {}'.format(feature_transformer.__class__))

        self.parameters = parameters
        self.feature_transformer = feature_transformer
        self.verbose = verbose

        self._a_syllable_penalty = -0.7
        self._noun_preference = 0.05

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
        # generate nodes and edges
        edges, bos_node, eos_node = self.parameters.generate(sentence)
        nodes = {node for edge in edges for node in edge[:2]}

        # add transition score
        edges = _hmm_style_tagger_weight(
            edges, self.parameters, self._a_syllable_penalty, self._noun_preference)

        # debug
        if debug:
            for from_, to_, score in edges:
                print('from : {}'.format(from_))
                print('to   : {}'.format(to_))
                print('score: {}\n'.format(score))

        # find optimal path
        list_of_eojeols, cost = ford_list(edges, nodes, bos_node, eos_node)

        # wrapper list of words to Eojeols
        eojeols = Eojeols(list_of_eojeols, cost)

        # post-processing
        if flatten:
            poses = _remain_only_pos(eojeols)
        else:
            poses = _remain_details(eojeols)

        return [poses, cost]

    def add_user_dictionary(self, tag, word_score):
        if not (tag in self.parameters.pos2words):
            raise ValueError('{} tag does not exist in model'.format(tag))
        for word, score in word_score.items():
            self.parameters.pos2words[tag][word] = score

########################################

class TrigramTagger(HMMStyleTagger):

    def __init__(self, parameters,
        feature_transformer=None, verbose=False):

        if feature_transformer is None:
            feature_transformer = TrigramFeatureTransformer()
        if parameters is None:
            parameters = TrigramFeatureTransformer()
        if verbose:
            print('use {}'.format(feature_transformer.__class__))

        self.parameters = parameters
        self.feature_transformer = feature_transformer
        self.verbose = verbose

        self._a_syllable_penalty = -0.3
        self._noun_preference = 0.5
        self._longer_noun_preference = 0.2

    def tag(self, sentence, flatten=True, beam_size=5):
        # generate nodes and edges
        begin_index = self.parameters.generate(sentence)

        # find optimal path
        chars = sentence.replace(' ', '')
        top_eojeols = beam_search(
            begin_index, beam_size, chars, self.parameters,
            self._a_syllable_penalty, self._noun_preference,
            self._longer_noun_preference
        )

        # post-processing
        def postprocessing(eojeols, flatten):
            return _remain_only_pos(eojeols) if flatten else _remain_details(eojeols)

        top_poses = [(postprocessing(eojeols, flatten), eojeols.score) for eojeols in top_eojeols]

        return top_poses

########################################
# common functions
def _remain_details(eojeols):
    return [(eojeol.pos, eojeol.begin, eojeol.end, eojeol.eojeol_score) for eojeol in eojeols.eojeols[1:-1]]

def _remain_only_pos(eojeols):
    poses = []
    for eojeol in eojeols.eojeols[1:-1]:
        poses.append((eojeol.first_word, eojeol.first_tag))
        if eojeol.is_compound:
            poses.append((eojeol.last_word, eojeol.last_tag))
    return poses