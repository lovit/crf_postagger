from ._beam import beam_search
from .. import AbstractTagger
from .. import TrigramFeatureTransformer

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

        super().__init__(parameters, feature_transformer, verbose)

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
            if flatten:
                return self._remain_only_pos(eojeols)
            else:
                return self._remain_details(eojeols)

        top_poses = [(postprocessing(eojeols, flatten), eojeols.score)
                     for eojeols in top_eojeols]

        return top_poses