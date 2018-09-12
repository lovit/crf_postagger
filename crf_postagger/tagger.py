import json
from .transformer import BaseFeatureTransformer
from .trainer import Feature

class TrainedCRFTagger:

    def __init__(self, model_path=None, coefficients=None,
        feature_transformer=None, verbose=False):

        if feature_transformer is None:
            feature_transformer = BaseFeatureTransformer()

        self.coef = coefficients
        self.feature_transformer = feature_transformer
        self.verbose = verbose
        if model_path:
            self._load_from_json(model_path)

    def score(self, sentence):

        # feature transform
        sentence_, tags = self.feature_transformer(sentence)

        # transition weight
        transitions = [(s0, s1) for s0, s1 in zip(tags, tags[1:])]
        score = sum((self.transitions.get(trans, 0) for trans in transitions))

        # state feature weight
        for features, tag in zip(sentence_, tags):
            for feature in features:
                score += self.state_features.get((feature, tag), 0)

        return score

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