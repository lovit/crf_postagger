from collections import namedtuple

Eojeol = namedtuple('Eojeol', 'pos first_word last_word first_tag last_tag begin end eojeol_score is_compound')
Eojeols = namedtuple('Eojeols', 'eojeols score')

bos = 'BOS'
eos = 'EOS'
unk = 'Unk'

BOS = Eojeol(bos, bos, bos, bos, bos, 0, 0, 0, 0)