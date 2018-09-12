try:
    import pycrfsuite
except:
    print('Failed to import python-crfsuite')

class Trainer:
    def __init__(self, potential_function, min_count=10, l2_cost=1.0, l1_cost=1.0):
        self.potential_function = potential_function
        self.min_count = min_count
        self.l2_cost = l2_cost
        self.l1_cost = l1_cost

    def scan_features(self, sentences):
        raise NotImplemented

    def train(self, sentences):

        features = self.scan_features(sentences)

        raise NotImplemented

def sent_to_xy(sentence, potential_function):
    raise NotImplemented