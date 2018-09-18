from .transformer import *

class TrainedCRFTagger:

    def __init__(self, node_generator,
        feature_transformer=None, verbose=False):

        if feature_transformer is None:
            feature_transformer = BaseFeatureTransformer()
        if node_generator is None:
            node_generator = HMMNodeGenerator()
        if verbose:
            print('use {}'.format(feature_transformer.__class__))

        self.feature_transformer = feature_transformer
        self.node_generator = node_generator
        self.verbose = verbose

    def score(self, wordpos_sentence, debug=False):

        # feature transform
        sentence_, tags = self.feature_transformer(wordpos_sentence)
        score = 0

        # transition weight
        for s0, s1 in zip(tags, tags[1:]):
            transition = (s0, s1)
            coef = self.node_generator.transitions.get(transition, 0)
            if debug:
                print('{} = {:f}, score = {:f}'.format(transition, coef, score))
            score += coef

        # state feature weight
        for features, tag in zip(sentence_, tags):
            for feature in features:
                if debug:
                    print('{} -> {} = {:f}, score = {:f}'.format(
                        feature, tag, coef, score))
                coef = self.node_generator.state_features.get((feature, tag), 0)
                score += coef

        return score

    def tag(self, sentence):
        edges, bos, eos = self.node_generator.generate(sentence)
        return edges

    def add_user_dictionary(self, tag, word_score):
        if not (tag in self.node_generator.pos2words):
            raise ValueError('{} tag does not exist in model'.format(tag))
        for word, score in word_score.items():
            self.node_generator.pos2words[tag][word] = score