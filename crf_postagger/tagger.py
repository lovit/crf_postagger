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

        self._a_syllable_penalty = -7

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
        # generate nodes and edges
        edges, bos_node, eos_node = self.node_generator.generate(sentence)
        nodes = {node for edge in edges for node in edge[:2]}

        # add transition score
        edges = self._add_weight(edges)

        # find optimal path
        path, cost = ford_list(edges, nodes, bos_node, eos_node)

        # post-processing
        path = self._postprocessing(path)

        return path

    def _add_weight(self, edges):
        def get_transition(f, t):
            return self.node_generator.transitions.get((f, t), 0)

        def get_score(from_, to_):
            score = get_transition(from_.last_tag, to_.first_tag) + to_.node_score
            if len(to_.first_word) == 1:
                score += self._a_syllable_penalty
            if not (to_.first_word == to_.last_tag):
                score += get_transition(to_.first_tag, to_.last_tag)
            return score
        return [(from_, to_, get_score(from_, to_)) for from_, to_ in edges]

    def _postprocessing(self, path):
        poses = []
        for node in path[1:-1]:
            poses_ = node.pos.split(' + ')
            if len(poses_) == 1:
                poses.append((node.first_word, node.first_tag))
            else:
                for pos in poses_:
                    word, tag = pos.rsplit('/', 1)
                    poses.append((word, tag))
        return poses

    def add_user_dictionary(self, tag, word_score):
        if not (tag in self.node_generator.pos2words):
            raise ValueError('{} tag does not exist in model'.format(tag))
        for word, score in word_score.items():
            self.node_generator.pos2words[tag][word] = score