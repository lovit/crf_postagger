feature_names = (
    'x[0]',
    'x[0]&y[-1]',
    'x[-1:0]',
    'x[-1:0]&y[-1]',
    'x[-1,1]',
    'x[-1,1]&y[-1]'
)
bos = 'BOS'
eos = 'EOS'

def potential_function(sentence):
    """Feature transformer
    
    :param list_of_tuple pos: a sentence [(word, tag), (word, tag), ...]
    """

    words, tags = zip(*sentence)
    words_ = tuple((bos, *words, eos))
    tags_ = tuple((bos, *tags, eos))

    n = len(sentence)
    sentence_ = [to_feature(words_, tags_, i) for i in range(1, n+1)]

    return sentence_

def to_feature(words_, tags_, i):
    features = [
        'x[0]=%s' % words_[i],
        'x[0]=%s, y[-1]=%s' % (words_[i], tags_[i-1]),
        'x[-1:0]=%s-%s' % (words_[i-1], words[i]),
        'x[-1:0]=%s-%s, y[-1]=%s' % (words_[i-1], words_[i], tags_[i-1]),
        'x[-1,1]=%s-%s' % (words_[i-1], words_[i+1]),
        'x[-1,1]=%s-%s, y[-1]=%s' % (words_[i-1], words_[i+1], tags_[i-1])
    ]
    return features