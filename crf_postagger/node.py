import re
doublespace_pattern = re.compile(u'\s+', re.UNICODE)

class AbstractNodeGenerator:
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
        return pos

    def _get_pos(self, word):
        return [tag for tag, words in self.pos2words.items() if word in words]