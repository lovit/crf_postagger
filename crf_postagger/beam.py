from collections import namedtuple
from .common import bos, eos, unk, BOS

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
            for pos in appending_poses:
                poses = (*immature.poses, pos)
                score = _trigram_beam_search_cumulate_score(
                    immature, pos, params, a_syllable_penalty,
                    noun_preference, longer_noun_preference)
                matures.append(Words(poses, score))
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
    word_prev, tag_prev = immature.poses[-1][:2]
    len_word = len(word)

    score = immature.score

    # preference & penalty
    score += (a_syllable_penalty * (1 + noun_preference * (tag == 'Noun')))  if len_word == 1 else 0
    score += noun_preference if (tag == 'Noun' and len_word > 1) else 0
    score += longer_noun_preference * (len(word) - 1) if tag == 'Noun' else 0

    # transition score
    score += params.transitions.get((tag_prev, tag), 0)

    # word feature
    score += params.pos2words.get(tag, {}).get(word, 0)

    # previous features
    score += params.previous_1X0.get(tag, {}).get((word_prev, word), 0)
    score += params.previous_X0_1Y.get(tag, {}).get((word, tag_prev), 0)

    # successive features (for previous pos)
    score += params.successive_X01.get(tag_prev, {}).get((word_prev, word), 0)
    score += params.successive_X01_Y1.get(tag_prev, {}).get((word_prev, word, tag), 0)

    # bothside features (for previous pos)
    if len(immature.poses) >= 2:
        word_prev2, tag_prev2 = immature.poses[-2][:2]
        score += params.bothside_1X1.get(tag_prev, {}).get((word_prev2, word), 0)
        score += params.bothside_1X01.get(tag_prev, {}).get((word_prev2, word_prev, word), 0)

    return score