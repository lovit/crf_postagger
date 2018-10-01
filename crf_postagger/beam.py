from collections import namedtuple
from .common import bos, eos, unk, BOS, Word

Words = namedtuple('Words', 'words score')

class Beam:
    def __init__(self, k):
        self.k = k
        self.beam = [[Words((BOS,), 0)]]

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

    def appending(immatures, appending_words, matures):
        for immature in immatures:
            for word in appending_words:
                words = (*immature.words, word)
                score = _trigram_beam_search_cumulate_score(
                    immature, word, params, a_syllable_penalty,
                    noun_preference, longer_noun_preference)
                matures.append(Words(words, score))
        return matures

    for e in range(1, len_sent + 1):
        matures = []

        for b in range(max(0, e - max_len), e):
            # prepare previous sequence
            immatures = beam[b]

            # prepare appending words
            appending_words = [word for word in begin_index[b] if word.end == e]

            if not appending_words:
                sub = chars[b:e]
                appending_words = [Word(sub+'/'+unk, sub, sub, unk, unk, b, e, 0, 0)]

            # appending
            matures = appending(immatures, appending_words, matures)

        # append beam and prune
        beam.append(matures)

    # for eos scoring
    EOS = Word('', eos, '', eos, '', len_sent, len_sent, 0, 0)
    matures = appending(beam[-1], [EOS], [])
    beam.append(matures)

    return beam[-1]

def _trigram_beam_search_cumulate_score(immature, word, params, a_syllable_penalty,
    noun_preference, longer_noun_preference):

    prev_word = immature.words[-1]
    len_word = word.end - word.begin

    score = immature.score + word.word_score

    # preference & penalty
    score += (a_syllable_penalty * (1 + noun_preference * (word.first_tag == 'Noun')))  if len_word == 1 else 0
    score += noun_preference if (word.first_tag == 'Noun' and len_word > 1) else 0
    score += longer_noun_preference * (len_word - 1) if word.first_tag == 'Noun' else 0

    # transition score
    score += params.transitions.get((prev_word.last_tag, word.first_tag), 0)

    # word feature
    # word score are already cumulated
    # score += params.pos2words.get(tag, {}).get(word, 0)

    # previous features
    score += params.previous_1X0.get(word.first_tag, {}).get((prev_word.last_word, word.first_word), 0)
    score += params.previous_X0_1Y.get(word.first_tag, {}).get((word.first_word, prev_word.last_tag), 0)

    # successive features (for previous pos)
    score += params.successive_X01.get(prev_word.last_tag, {}).get((prev_word.last_word, word.first_word), 0)
    score += params.successive_X01_Y1.get(prev_word.last_tag, {}).get((prev_word.last_word, word.first_word, word.first_tag), 0)

    # bothside features (for previous pos)
    if len(immature.words) >= 2:
        prev2_word = immature.words[-2]
        score += params.bothside_1X1.get(prev_word.first_tag, {}).get((prev2_word.last_word, word.first_word), 0)
        score += params.bothside_1X01.get(prev_word.first_tag, {}).get((prev2_word.last_word, prev_word.first_word, word.first_word), 0)

    return score