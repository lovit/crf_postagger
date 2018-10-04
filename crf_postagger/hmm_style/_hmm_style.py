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