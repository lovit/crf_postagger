import json
from .transformer import BaseFeatureTransformer
from .trainer import Feature

class TrainedCRFTagger:

    def __init__(self, model_path=None, coefficients=None,
        feature_transformer=None, verbose=False):

        if feature_transformer is None:
            feature_transformer = BaseFeatureTransformer()
        if verbose:
            print('use {}'.format(feature_transformer.__class__))

        self.coef = coefficients
        self.feature_transformer = feature_transformer
        self.verbose = verbose
        if model_path:
            self._load_from_json(model_path)

    def score(self, sentence, debug=False):

        # feature transform
        sentence_, tags = self.feature_transformer(sentence)
        score = 0

        # transition weight
        for s0, s1 in zip(tags, tags[1:]):
            transition = (s0, s1)
            coef = self.transitions.get(transition, 0)
            if debug:
                print('{} = {:f}, score = {:f}'.format(transition, coef, score))
            score += coef

        # state feature weight
        for features, tag in zip(sentence_, tags):
            for feature in features:
                if debug:
                    print('{} -> {} = {:f}, score = {:f}'.format(
                        feature, tag, coef, score))
                coef = self.state_features.get((feature, tag), 0)
                score += coef

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