from collections import namedtuple

Word = namedtuple('Word', 'pos first_word last_word first_tag last_tag begin end word_score is_compound')
Words = namedtuple('Words', 'words score')

bos = 'BOS'
eos = 'EOS'
unk = 'Unk'

BOS = Word(bos, bos, bos, bos, bos, 0, 0, 0, 0)