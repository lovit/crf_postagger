from .transformer import *
from .path import ford_list

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

        self._a_syllable_penalty = -7
        self._noun_preference = 10

    def score(self, wordpos_sentence, debug=False):

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

    def tag(self, sentence, debug=False):
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
        path, cost = ford_list(edges, nodes, bos_node, eos_node)

        # post-processing
        path = self._postprocessing(path)

        return path

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
        if not (tag in self.parameters.pos2words):
            raise ValueError('{} tag does not exist in model'.format(tag))
        for word, score in word_score.items():
            self.parameters.pos2words[tag][word] = score

def _hmm_style_tagger_weight(edges, parameters, _a_syllable_penalty, _noun_preference):
    def get_transition(f, t):
        return parameters.transitions.get((f, t), 0)

    def get_score(from_, to_):
        #score = get_transition(from_.last_tag, to_.first_tag) + to_.node_score
        score = get_transition(from_.last_tag, to_.first_tag) + from_.node_score + to_.node_score
        if len(to_.first_word) == 1:
            score += _a_syllable_penalty
        #if not (to_.first_word == to_.last_tag):
        #    score += get_transition(to_.first_tag, to_.last_tag)
        if to_.first_tag == 'Noun':
            score += _noun_preference
        return score
    return [(from_, to_, get_score(from_, to_)) for from_, to_ in edges]

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

        self._a_syllable_penalty = -7

    def tag(self, sentence, debug=False, k=5):
        # generate nodes and edges
        begin_index = self.parameters.generate(sentence)
        end_index = _to_end_index(begin_index)

        # find optimal path
        paths = _trigram_tagger_beam_search(begin_index, end_index, k)

        # post-processing
        paths = [self._postprocessing(path) for path in paths]

        return paths

def _trigram_tagger_beam_search(begin_index, end_index, k):
    raise NotImplemented

def _to_end_index(begin_index):
    end_index = [[] for _ in range(len(begin_index) + 1)]
    for words in begin_index:
        for word in words:
            # format: (word, tag, b, e)
            end_index[word[3]].append(word)
    return end_index