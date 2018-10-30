from collections import namedtuple
import os
import psutil


Eojeol = namedtuple('Eojeol', 'pos first_word last_word first_tag last_tag begin end eojeol_score compound unknown')
Eojeols = namedtuple('Eojeols', 'eojeols score')

bos = 'BOS'
eos = 'EOS'
unk = 'Unk'

BOS = Eojeol(bos, bos, bos, bos, bos, 0, 0, 0, 0, 0)

installpath = os.path.sep.join(
    os.path.dirname(os.path.realpath(__file__)).split(os.path.sep)[:-1])

class Corpus:
    def __init__(self, path, num_sent=-1):
        self.path = path
        self.num_sent = num_sent
    def __iter__(self):
        with open(self.path, encoding='utf-8') as f:
            for i, sent in enumerate(f):
                if self.num_sent > 0 and i > self.num_sent:
                    break
                morphtags = [token.rsplit('/', 1) for token in sent.split()]
                morphtags = [token for token in morphtags if len(token) == 2]
                if morphtags:
                    yield morphtags

def _to_end_index(begin_index):
    end_index = [[] for _ in range(len(begin_index) + 1)]
    for words in begin_index:
        for word in words:
            # format: (word, tag, b, e)
            end_index[word[3]].append(word)
    return end_index

def check_dirs(path):
    dirname = os.path.dirname(path)
    if dirname and dirname != '.' and not os.path.exists(dirname):
        os.makedirs(dirname)

def get_process_memory():
    """It returns the memory usage of current process"""

    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3)