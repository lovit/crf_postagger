from .. import AbstractTagger
from .. import AbstractParameter
from .. import BaseFeatureTransformer
from .. import bos, eos, unk, Eojeol, Eojeols
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

def _hmm_style_tagger_weight(edges, parameters, _a_syllable_penalty, _noun_preference):
    def get_transition(f, t):
        return parameters.transitions.get((f, t), 0)

    def get_score(from_, to_):
        #score = get_transition(from_.last_tag, to_.first_tag) + to_.eojeol_score
        score = get_transition(from_.last_tag, to_.first_tag) + from_.eojeol_score + to_.eojeol_score
        if len(to_.first_word) == 1:
            score += _a_syllable_penalty
        elif to_.first_tag == 'Noun':
            score += _noun_preference
        #if not (to_.first_word == to_.last_tag):
        #    score += get_transition(to_.first_tag, to_.last_tag)
        return score
    return [(from_, to_, get_score(from_, to_)) for from_, to_ in edges]

class HMMStyleParameter(AbstractParameter):
    def __init__(self, model_path=None, pos2words=None, preanalyzed_eojeols=None,
        max_word_len=0, parameter_marker=' -> '):

        super().__init__(model_path, pos2words,
            preanalyzed_eojeols, max_word_len, parameter_marker)

    def generate(self, sentence):
        # prepare lookup list
        chars = sentence.replace(' ','')
        sent = self._sentence_lookup(sentence)
        n_char = len(sent) + 1

        # add end node
        eos_node = Eojeol(eos, eos, None, eos, None, n_char-1, n_char, 0, 0)
        sent.append([eos_node])

        # check first word position
        nonempty_first = self._get_nonempty_first(sent, n_char)
        if nonempty_first > 0:
            # (pos, first_word, last_word, first_tag, last_tag, begin, end, eojeol_score, is_compound)
            word = chars[:nonempty_first]
            sent[0] = [Eojeol(word, word, word, unk, unk, 0, nonempty_first, 0, 0)]

        # add link between adjacent nodes
        edges = self._link_adjacent_nodes(sent, chars, n_char)

        # add link from unk node
        edges = self._link_from_unk_nodes(edges, sent)

        bos_node = Eojeol(bos, None, bos, None, bos, 0, 0, 0, 0)
        for word in sent[0]:
            edges.append((bos_node, word))
        edges = sorted(edges, key=lambda x:(x[0].begin, x[1].end))

        return edges, bos_node, eos_node

    def _get_nonempty_first(self, sent, end, offset=0):
        for i in range(offset, end):
            if sent[i]:
                return i
        return offset

    def _link_adjacent_nodes(self, sent, chars, n_char):
        edges = []
        for words in sent[:-1]:
            for word in words:
                if not sent[word.end]:
                    unk_end = self._get_nonempty_first(sent, n_char, word.end)
                    unk_word = chars[word.end:unk_end]
                    unk_node = Eojeol(unk_word, unk_word, unk_word, unk, unk, word.end, unk_end, 0)
                    edges.append((word, unk_node))
                for adjacent in sent[word.end]:
                    edges.append((word, adjacent))
        return edges

    def _link_from_unk_nodes(self, edges, sent):
        unk_nodes = {to_node for _, to_node in edges if to_node.last_tag == unk}
        for unk_node in unk_nodes:
            for adjacent in sent[unk_node.end]:
                edges.append((unk_node, adjacent))
        return edges