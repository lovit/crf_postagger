from collections import defaultdict
from collections import namedtuple
import re
import json

from .common import bos, eos, unk
from .lemmatizer import lemma_candidate
from .trainer import Feature

doublespace_pattern = re.compile(u'\s+', re.UNICODE)
Word = namedtuple('Word', 'pos first_word last_word first_tag last_tag begin end node_score')

class AbstractParameter:
    def __init__(self, model_path=None, pos2words=None,
        max_word_len=0, parameter_marker=' -> '):

        self.pos2words = pos2words
        self.max_word_len = max_word_len

        if model_path:
            self._load_from_json(model_path, parameter_marker)

        if not pos2words:
            self._construct_dictionary_from_state_features()

        if self.max_word_len == 0:
            self._check_max_word_len()

    def __call__(self, sentence):
        return self.generate(sentence)

    def generate(self, sentence):
        return self._sentence_lookup(sentence)

    def _check_max_word_len(self):
        if not self.pos2words:
            raise ValueError('pos2words should not be empty')
        self.max_word_len = max(
            max(len(word) for word in words) for words in self.pos2words.values())

    def _sentence_lookup(self, sentence):
        sentence = doublespace_pattern.sub(' ', sentence)
        sent = []
        for eojeol in sentence.split():
            sent += self._word_lookup(eojeol, offset=len(sent))
        return sent

    def _word_lookup(self, eojeol, offset=0):
        n = len(eojeol)
        pos = [[] for _ in range(n)]
        for b in range(n):
            for r in range(1, self.max_word_len+1):
                e = b+r
                if e > n:
                    continue
                sub = eojeol[b:e]
                for tag in self._get_pos(sub):
                    pos[b].append((sub, tag, b+offset, e+offset))
                for lemma_node in self._add_lemmas(sub, r, b, e, offset):
                    pos[b].append(lemma_node)
        return pos

    def _get_pos(self, word):
        return tuple(tag for tag, words in self.pos2words.items() if word in words)

    def _add_lemmas(self, sub, r, b, e, offset):
        for i in range(1, min(self.max_word_len, len(sub)) + 1):
            try:
                for l_morph, r_morph, l_tag, r_tag in self._lemmatize(sub, i):
                    node = ('%s + %s' %  (l_morph, r_morph),
                            '%s + %s' % (l_tag, r_tag),
                            b + offset, e + offset)
                    yield node
            except Exception as e:
                #print(e)
                continue

    def _lemmatize(self, word, i):
        l = word[:i]
        r = word[i:]
        lemmas = []
        len_word = len(word)
        for l_, r_ in lemma_candidate(l, r):
            if (l_ in self.pos2words.get('Verb', {})) and (r_ in self.pos2words.get('Eomi', {})):
                yield (l_, r_, 'Verb', 'Eomi')
            if (l_ in self.pos2words.get('Adjective', {})) and (r_ in self.pos2words.get('Eomi', {})):
                yield (l_, r_, 'Adjective', 'Eomi')
            if len_word > 1 and not (word in self.pos2words.get('Noun', {})):
                if (l_ in self.pos2words['Noun']) and ( (r_ == 'ㄴ') or (r_ == 'ㄹ') ):
                    yield (l_, r_, 'Noun', 'Josa')

    def _load_from_json(self, json_path, marker = ' -> '):
        with open(json_path, encoding='utf-8') as f:
            model = json.load(f)

        # parse transition
        self.transitions = {
            tuple(trans.split(marker)): coef
            for trans, coef in model['transitions'].items()
        }

        # parse state features
        self.state_features = {
            tuple(feature.split(marker)): coef
            for feature, coef in model['state_features'].items()
        }

        # weight normalize. [-1, 1]
        max_value = max(abs(coef) for coef in self.transitions.values())
        max_value = max(max_value, max(abs(coef) for coef in self.state_features.values()))
        self.transitions = {key:coef/max_value for key, coef in self.transitions.items()}
        self.state_features = {key:coef/max_value for key, coef in self.state_features.items()}

        # get idx2features
        self.idx2feature = model['idx2feature']

        # parse feature information map
        self.features = {
            feature: Feature(idx, count)
            for feature, (idx, count) in model['features'].items()
        }

        del model

    def _construct_dictionary_from_state_features(self):
        self.pos2words = defaultdict(lambda: {})
        for (feature, tag), coef in self.state_features.items():
            if (feature[:4] == 'x[0]') and not (', ' in feature) and coef > 0:
                word = feature[5:]
                self.pos2words[tag][word] = coef
        self.pos2words = dict(self.pos2words)

class HMMStyleParameter(AbstractParameter):
    def __init__(self, model_path=None, pos2words=None,
        max_word_len=0, parameter_marker=' -> '):

        super().__init__(
            model_path, pos2words, max_word_len, parameter_marker)

    def generate(self, sentence):
        # prepare lookup list
        chars = sentence.replace(' ','')
        sent = self._sentence_lookup(sentence)
        n_char = len(sent) + 1

        # wrap node to Word
        def to_Word(node):
            words, tags, b, e = node
            words = words.split(' + ')
            tags = tags.split(' + ')
            score = self.pos2words.get(tags[0], {}).get(words[0], 0)
            if len(words) == 1:
                return Word(words[0], words[0], words[0], tags[0], tags[0], b, e, score)
            pos = '%s/%s + %s/%s' % (words[0], tags[0], words[1], tags[1])
            score += self.pos2words.get(tags[1], {}).get(words[1], 0)
            return Word(pos, words[0], words[1], tags[0], tags[1], b, e, score)

        sent = [[to_Word(node) for node in words] for words in sent]

        # add end node
        eos_node = Word(eos, eos, None, eos, None, n_char-1, n_char, 0)
        sent.append([eos_node])

        # check first word position
        nonempty_first = self._get_nonempty_first(sent, n_char)
        if nonempty_first > 0:
            # (words, first_word, last_word, first_tag, last_tag, begin, end, node_score)
            word = chars[:nonempty_first]
            sent[0] = [Word(word, word, word, unk, unk, 0, nonempty_first, 0)]

        # add link between adjacent nodes
        edges = self._link_adjacent_nodes(sent, chars, n_char)

        # add link from unk node
        edges = self._link_from_unk_nodes(edges, sent)

        bos_node = Word(bos, None, bos, None, bos, 0, 0, 0)
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
                    unk_node = Word(unk_word, unk_word, unk_word, unk, unk, word.end, unk_end, 0)
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

class TrigramParameter(AbstractParameter):
    def __init__(self, model_path=None, pos2words=None,
        max_word_len=0, parameter_marker=' -> '):

        super().__init__(
            model_path, pos2words, max_word_len, parameter_marker)

        self._separate_features()

    def _separate_features(self):
        is_1X0 =    lambda x: ('x[-1:0]' in x) and not (' ' in x)
        is_X0_1Y =  lambda x: ('y[-1]' in x) and not (' ' in x)
        is_X01 =    lambda x: ('x[0:1]') in x and not (' ' in x)
        is_X01_Y1 = lambda x: ('x[0:1]') in x and ('y[1]' in x)
        is_1X1 =    lambda x: ('x[-1,1]' in x)
        is_1X01 =   lambda x: ('x[-1:1]' in x)

        def parse_word(feature):
            poses = feature.split(', ')
            wordtags = tuple(
                wordtag for pos in poses for wordtag
                in pos.split(']=')[-1].split('-')
            )
            return wordtags

        # previous features
        self.previous_1X0 = defaultdict(lambda: {})
        self.previous_X0_1Y = defaultdict(lambda: {})

        # successive features
        self.successive_X01 = defaultdict(lambda: {})
        self.successive_X01_Y1 = defaultdict(lambda: {})

        # bothside_features
        self.bothside_1X1 = defaultdict(lambda: {})
        self.bothside_1X01 = defaultdict(lambda: {})

        for (feature, tag), coef in self.state_features.items():
            if is_1X0(feature):
                self.previous_1X0[tag][parse_word(feature)] = coef
            elif is_X0_1Y(feature):
                self.previous_X0_1Y[tag][parse_word(feature)] = coef
            elif is_X01(feature):
                self.successive_X01[tag][parse_word(feature)] = coef
            elif is_X01_Y1(feature):
                self.successive_X01_Y1[tag][parse_word(feature)] = coef
            elif is_1X1(feature):
                self.bothside_1X1[tag][parse_word(feature)] = coef
            elif is_1X01(feature):
                self.bothside_1X01[tag][parse_word(feature)] = coef

        # previous features
        self.previous_1X0 = dict(self.previous_1X0)
        self.previous_X0_1Y = dict(self.previous_X0_1Y)
        # successive features
        self.successive_X01 = dict(self.successive_X01)
        self.successive_X01_Y1 = dict(self.successive_X01_Y1)
        # bothside_features
        self.bothside_1X1 = dict(self.bothside_1X1)
        self.bothside_1X01 = dict(self.bothside_1X01)

    #def generate(self, sentence):
    #    raise NotImplemented