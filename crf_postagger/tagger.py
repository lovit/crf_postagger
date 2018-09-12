from .transformer import potential_function as default_potential_function

class TrainedCRFTagger:

    def __init__(self, coefficients=None, potential_function=None):
        self.coef = coefficients

        if potential_function is None:
            potential_function = default_potential_function
        self.potential_function = potential_function

    def score(self, features):
        raise NotImplemented

    def tag(self, sentence):
        raise NotImplemented