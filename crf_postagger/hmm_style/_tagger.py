from .. import AbstractTagger
from .. import BaseFeatureTransformer
from .. import bos, eos, unk, Eojeols
from ._hmm_style import _hmm_style_tagger_weight
from ._path import ford_list

class HMMStyleTagger(AbstractTagger):
    def __init__(self, parameters,
        feature_transformer=None, verbose=False):

        if feature_transformer is None:
            feature_transformer = BaseFeatureTransformer()
        if parameters is None:
            parameters = HMMStyleParameter()

        self._a_syllable_penalty = -0.7
        self._noun_preference = 0.05

        super().__init__(parameters, feature_transformer, verbose)

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
            poses = self._remain_only_pos(eojeols)
        else:
            poses = self._remain_details(eojeols)

        return [poses, cost]