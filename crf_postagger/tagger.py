import json
from .transformer import potential_function as default_potential_function
from .trainer import Feature

class TrainedCRFTagger:

    def __init__(self, model_path=None, coefficients=None,
        potential_function=None, verbose=False):

        if potential_function is None:
            potential_function = default_potential_function

        self.coef = coefficients
        self.potential_function = potential_function
        self.verbose = verbose
        if model_path:
            self._load_from_json(model_path)

    def score(self, features):
        raise NotImplemented

    def tag(self, sentence):
        raise NotImplemented

    def _load_from_json(self, json_path, marker = ' -> '):
        with open(json_path, encoding='utf-8') as f:
            model = json.load(f)

        # parse transition
        self.transitions = {
            tuple(trans.split(marker)): coef
            for trans, coef in model['transitions'].items()
        }

        # parse state features
        self.state_features = {
            tuple(feature.split(marker)): coef
            for feature, coef in model['state_features'].items()
        }

        # get idx2features
        self._idx2feature = model['idx2feature']

        # parse feature information map
        self._features = {
            feature: Feature(idx, count)
            for feature, (idx, count) in model['features'].items()
        }

        del model