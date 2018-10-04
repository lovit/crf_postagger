from .. import bos, eos, unk, BOS, Eojeol, Eojeols

class Beam:
    def __init__(self, k):
        self.k = k
        self.beam = [[Eojeols((BOS,), 0)]]

    def __getitem__(self, index):
        return self.beam[index]

    def append(self, candidates):
        # descending order of score, last item in list
        candidates = sorted(candidates, key=lambda x:-x.score)[:self.k]
        self.beam += [candidates]

def beam_search(begin_index, k, chars, params, score_functions,
                unknown_penalty, **kwargs):

    len_sent = len(chars)
    max_len = params.max_word_len
    beam = Beam(k)

    def appending(immatures, appending_words, matures):
        for immature in immatures:
            for eojeol in appending_words:
                eojeols = (*immature.eojeols, eojeol)
                score = immature.score
                for func in score_functions:
                    score += func(immature, eojeol, params, **kwargs)
                matures.append(Eojeols(eojeols, score))
        return matures

    for e in range(1, len_sent + 1):
        matures = []

        for b in range(max(0, e - max_len), e):
            # prepare previous sequence
            immatures = beam[b]

            # prepare appending words
            appending_eojeols = [eojeol for eojeol in begin_index[b] if eojeol.end == e]

            if not appending_eojeols:
                sub = chars[b:e]
                appending_eojeols = [Eojeol(sub+'/'+unk, sub, sub, unk, unk, b, e, unknown_penalty, 0)]

            # appending
            matures = appending(immatures, appending_eojeols, matures)

        # append beam and prune
        beam.append(matures)

    # for eos scoring
    EOS = Eojeol('', eos, '', eos, '', len_sent, len_sent, 0, 0)
    matures = appending(beam[-1], [EOS], [])
    beam.append(matures)

    return beam[-1]

def _preference_penalty(immature, eojeol, params, a_syllable_penalty,
    noun_preference, longer_noun_preference):

    len_eojeol = eojeol.end - eojeol.begin
    score = (a_syllable_penalty * (1 + noun_preference * (eojeol.first_tag == 'Noun')))  if len_eojeol == 1 else 0
    score += noun_preference if (eojeol.first_tag == 'Noun' and len_eojeol > 1) else 0
    score += longer_noun_preference * (len_eojeol - 1) if eojeol.first_tag == 'Noun' else 0
    return score

def _trigram_score(immature, eojeol, params, **kargs):

    eojeol_prev = immature.eojeols[-1]
    # eojeol score, x[0]
    score = eojeol.eojeol_score

    # transition score
    score += params.transitions.get((eojeol_prev.last_tag, eojeol.first_tag), 0)

    # previous features
    score += params.previous_1X0.get(eojeol.first_tag, {}).get((eojeol_prev.last_word, eojeol.first_word), 0)
    score += params.previous_X0_1Y.get(eojeol.first_tag, {}).get((eojeol.first_word, eojeol_prev.last_tag), 0)

    # successive features (for previous pos)
    score += params.successive_X01.get(eojeol_prev.last_tag, {}).get((eojeol_prev.last_word, eojeol.first_word), 0)
    score += params.successive_X01_Y1.get(eojeol_prev.last_tag, {}).get((eojeol_prev.last_word, eojeol.first_word, eojeol.first_tag), 0)

    # bothside features (for previous pos)
    if len(immature.eojeols) >= 2:
        eojeol_prev2 = immature.eojeols[-2]
        score += params.bothside_1X1.get(eojeol_prev.first_tag, {}).get((eojeol_prev2.last_word, eojeol.first_word), 0)
        score += params.bothside_1X01.get(eojeol_prev.first_tag, {}).get((eojeol_prev2.last_word, eojeol_prev.first_word, eojeol.first_word), 0)

    return score