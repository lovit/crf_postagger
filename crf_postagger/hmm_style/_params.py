from .. import AbstractParameter
from .. import Eojeol
from .. import eos, bos, unk

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