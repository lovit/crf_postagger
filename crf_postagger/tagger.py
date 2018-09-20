from collections import namedtuple
from .common import bos, eos, unk
from .transformer import *
from .path import ford_list
from .utils import _to_end_index

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
        # TODO: common unitfy or separate
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

        self._a_syllable_penalty = -3
        self._noun_preference = 5
        self._longer_noun_preference = 2

    def tag(self, sentence, debug=False, k=5):
        # generate nodes and edges
        begin_index = self.parameters.generate(sentence)

        # find optimal path
        chars = sentence.replace(' ', '')
        paths = beam_search(
            begin_index, k, chars, self.parameters,
            self._a_syllable_penalty, self._noun_preference,
            self._longer_noun_preference
        )

        # post-processing
        #paths = [self._postprocessing(path) for path in paths]
        paths = [(path.poses[1:-1], path.score) for path in paths]

        return paths


Poses = namedtuple('Poses', 'poses score')

class Beam:
    def __init__(self, k):
        self.k = k
        self.beam = [[Poses(((bos, bos, 0, 0),), 0)]]

    def __getitem__(self, index):
        return self.beam[index]

    def append(self, candidates):
        # descending order of score, last item in list
        candidates = sorted(candidates, key=lambda x:-x.score)[:self.k]
        self.beam += [candidates]

def beam_search(begin_index, k, chars, params,
    a_syllable_penalty, noun_preference, longer_noun_preference):

    len_sent = len(chars)
    max_len = params.max_word_len
    beam = Beam(k)

    def appending(immatures, appending_poses, matures):
        for immature in immatures:
            for pos in appending_poses:
                poses = (*immature.poses, pos)
                score = _trigram_beam_search_cumulate_score(
                    immature, pos, params, a_syllable_penalty,
                    noun_preference, longer_noun_preference)
                matures.append(Poses(poses, score))
        return matures

    for e in range(1, len_sent + 1):
        matures = []

        for b in range(max(0, e - max_len), e):
            # prepare previous sequence
            immatures = beam[b]

            # prepare appending poses
            appending_poses = [pos for pos in begin_index[b] if pos[3] == e]

            if not appending_poses:
                appending_poses = [(chars[b:e], unk, b, e)]

            # appending
            matures = appending(immatures, appending_poses, matures)

        # append beam and prune
        beam.append(matures)

    # for eos scoring
    matures = appending(beam[-1], [(eos, eos, len_sent, len_sent)], [])
    beam.append(matures)

    return beam[-1]

def _trigram_beam_search_cumulate_score(immature, pos, params, a_syllable_penalty,
    noun_preference, longer_noun_preference):

    word, tag = pos[:2]
    word_l, tag_l = immature.poses[-1][:2]

    score = immature.score

    # preference & penalty
    score += (a_syllable_penalty * (1 + noun_preference * (tag == 'Noun')))  if len(word) == 1 else 0
    score += noun_preference if tag == 'Noun' else 0
    score += longer_noun_preference * (len(word) - 1) if tag == 'Noun' else 0

    # transition score
    score += params.transitions.get((tag_l, tag), 0)

    # word feature
    score += params.pos2words.get(tag, {}).get(word, 0)

    # previous features
    score += params.previous_1X0.get(tag, {}).get((word_l, word), 0)
    score += params.previous_X0_1Y.get(tag, {}).get((word, tag_l), 0)

    # successive features (for previous pos)
    score += params.successive_X01.get(tag_l, {}).get((word_l, word), 0)
    score += params.successive_X01_Y1.get(tag_l, {}).get((word_l, word, tag), 0)

    # bothside features (for previous pos)
    if len(immature.poses) >= 2:
        word_ll, tag_ll = immature.poses[-2][:2]
        score += params.bothside_1X1.get(tag_l, {}).get((word_ll, word), 0)
        score += params.bothside_1X01.get(tag_l, {}).get((word_ll, word_l, word), 0)

    return score