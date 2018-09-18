import re
from .lemmatizer import lemma_candidate

doublespace_pattern = re.compile(u'\s+', re.UNICODE)

class BaseNodeGenerator:
    def __init__(self, pos2words, state_features, max_word_len=0):
        self.pos2words = pos2words
        self.state_features = state_features
        self.max_word_len = max_word_len

        if self.max_word_len == 0:
            self._check_max_word_len()

    def _check_max_word_len(self):
        if not self.pos2words:
            raise ValueError('pos2words should not be empty')
        self.max_word_len = max(
            max(len(word) for word in words) for words in self.pos2words.values())

    def generate(self, sentence):
        raise NotImplemented

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
                    pos[b].append((sub, tag, tag, b+offset, e+offset))
                for lemma_node in self._add_lemmas(sub, r, b, e, offset):
                    pos[b].append(lemma_node)
        return pos

    def _get_pos(self, word):
        return [tag for tag, words in self.pos2words.items() if word in words]

    def _add_lemmas(self, sub, r, b, e, offset):
        for i in range(1, min(self.max_word_len, len(sub)) + 1):
            try:
                lemmas = self._lemmatize(sub, i)
                for sub_, tag0, tag1 in lemmas:
                    node = (sub_, tag0, tag1, b+offset, e+offset)
                    yield node
            except:
                continue

    def _lemmatize(self, word, i):
        l = word[:i]
        r = word[i:]
        lemmas = []
        len_word = len(word)
        for l_, r_ in lemma_candidate(l, r):
            word_ = l_ + ' + ' + r_
            if (l_ in self.pos2words.get('Verb', {})) and (r_ in self.pos2words.get('Eomi', {})):
                lemmas.append((word_, 'Verb', 'Eomi'))
            if (l_ in self.pos2words.get('Adjective', {})) and (r_ in self.pos2words.get('Eomi', {})):
                lemmas.append((word_, 'Adjective', 'Eomi'))
            if len_word > 1 and not (word in self.pos2words.get('Noun', {})):
                if (l_ in self.pos2words['Noun']) and (r_ in self.pos2words['Josa']):
                    lemmas.append((word_, 'Noun', 'Josa'))
        return lemmas