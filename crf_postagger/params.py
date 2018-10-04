from collections import defaultdict
import re
import json

from .utils import bos, eos, unk, Eojeol
from .lemmatizer import lemma_candidate
from .trainer import Feature

doublespace_pattern = re.compile(u'\s+', re.UNICODE)

class AbstractParameter:
    def __init__(self, model_path=None, pos2words=None, preanalyzed_eojeols=None,
        max_word_len=0, parameter_marker=' -> ', unknown_penalty=-0.01):

        self.pos2words = pos2words
        self.max_word_len = max_word_len
        self.unknown_penalty = unknown_penalty

        if model_path:
            self._load_from_json(model_path, parameter_marker)

        if not pos2words:
            self._construct_dictionary_from_state_features()

        if self.max_word_len == 0:
            self._check_max_word_len()

        if not preanalyzed_eojeols:
            preanalyzed_eojeols = {}
        self.preanalyzed_eojeols = preanalyzed_eojeols
        self._update_dictionary_with_preanalyzed_eojeols()

    def __call__(self, sentence):
        return self.generate(sentence)

    def generate(self, sentence, guess_tag=False):
        return self._sentence_lookup(sentence, guess_tag)

    def _check_max_word_len(self):
        if not self.pos2words:
            raise ValueError('pos2words should not be empty')
        self.max_word_len = max(
            max(len(word) for word in words) for words in self.pos2words.values())

    def _sentence_lookup(self, sentence, guess_tag=False):
        sentence = doublespace_pattern.sub(' ', sentence)
        sent = []
        for eojeol in sentence.split():
            sent += self._word_lookup(eojeol, len(sent), guess_tag)
        return sent

    def _word_lookup(self, eojeol, offset=0, guess_tag=False):
        n = len(eojeol)
        pos = [[] for _ in range(n)]
        for b in range(n):
            for r in range(1, self.max_word_len+1):
                e = b+r
                if e > n:
                    continue
                sub = eojeol[b:e]

                # Eojeol(pos, first_word, last_word, first_tag, last_tag, begin, end, eojeol_score, compound, unknown)
                tag_scores = self._get_tag_score(sub)

                # when substring is known word
                if tag_scores:
                    for tag, score in tag_scores:
                        pos[b].append(Eojeol(sub+'/'+tag, sub, sub, tag, tag, b+offset, e+offset, score, 0, 0))
                # when substring is unknown substring
                elif guess_tag:
                    for tag, score in self._guess_tag(sub, b, e, eojeol):
                        pos[b].append(Eojeol(sub+'/'+tag, sub, sub, tag, tag, b+offset, e+offset, score, 0, 1))

                # check whether substring is predicator
                for word_form_lemma in self._add_lemmas(sub, b, e, offset):
                    pos[b].append(word_form_lemma)

        return pos

    def _get_tag_score(self, word):
        # return (word, word score)
        return tuple((tag, words[word]) for tag, words in self.pos2words.items() if word in words)

    def _guess_tag(self, sub, b, e, eojeol):
        return [
            ('Noun', self.unknown_penalty),
            ('Adverb', self.unknown_penalty),
            ('Exclamation', self.unknown_penalty),
            (unk, self.unknown_penalty)
        ]

    def _add_lemmas(self, sub, b, e, offset):

        def get_score(word, tag):
            return self.pos2words.get(tag, {}).get(word, 0)

        def as_eojeol(l_morph, r_morph, l_tag, r_tag, b, e, offset):
            eojeol = Eojeol(
                '%s/%s + %s/%s' %  (l_morph, l_tag, r_morph, r_tag),
                l_morph, r_morph, l_tag, r_tag, b + offset, e + offset,
                get_score(l_morph, l_tag) + get_score(r_morph, r_tag), 1, 0
            )
            return eojeol

        # check pre-analyzed lemmas
        lemmas = self.preanalyzed_eojeols.get(sub, [])

        # if sub is unseen string
        if not lemmas:
            for i in range(1, min(self.max_word_len, len(sub)) + 1):
                try:
                    for lemma in self._lemmatize(sub, i):
                        lemmas.append(lemma)
                except Exception as e:
                    continue

        # formatting
        lemmas = [as_eojeol(l_morph, r_morph, l_tag, r_tag, b, e, offset)
                  for l_morph, r_morph, l_tag, r_tag in lemmas]

        return lemmas

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

    def _update_dictionary_with_preanalyzed_eojeols(self):
        for preanalyzeds in self.preanalyzed_eojeols.values():
            for l_morph, r_morph, l_tag, r_tag in preanalyzeds:
                if l_tag in self.pos2words:
                    self.pos2words[l_tag][l_morph] = max(
                        0, self.pos2words[l_tag].get(l_morph, 0))
                if r_tag in self.pos2words:
                    self.pos2words[r_tag][r_morph] = max(
                        0, self.pos2words[r_tag].get(r_morph, 0))