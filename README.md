## Conditional Random Field (CRF) 기반 한국어 형태소 분석기

세종 말뭉치를 이용하여 학습한 CRF 기반 한국어 형태소 분석기입니다. CRF 을 이용하여 형태소 분석을 하는 과정을 설명하기 위한 코드입니다.

학습을 위해서는 crfsuite 의 Python wrapper 인 python-crfsuite 를 이용하였습니다. Decoding 과 tagging 과정은 Python 코드로만 이뤄져 있습니다. python-crfsuite 의 tagging 기능을 이용하지 않습니다.

이 repository 에서는 CRF 모델을 학습하는 Trainer 와, 학습된 모델을 이용하여 품사 판별을 하는 HMMStyleTagger 및 TrigramTagger 을 제공합니다.

구현 과정 및 원리에 대해서는 [블로그 포스트][crf_tagger_post]를 참고하세요.

## Usage

### Training

학습을 위하여 Corpus 와 CorpusTrainer class 를 import 합니다. num_sent 에 원하는 샘플 문장의 숫자를 입력할 수 있습니다. 기본값은 -1 이며, corpus 의 모든 문장을 yield 합니다.

    from crf_postagger import Corpus

    corpus_path = '../data/sejong_simpletag.txt'
    corpus = Corpus(corpus_path, num_sent=3)

Corpus 는 nested list 형식의 문장을 yield 하는 class 입니다. 학습에 이용한 네 문장의 예시입니다. 각 문장은 list 로 표현되며, 문장은 [형태소, 품사] 의 list 로 구성되어 있습니다.

    for sentence in corpus:
        print(sentence)

    [['뭐', 'Noun'], ['타', 'Verb'], ['고', 'Eomi'], ['가', 'Verb'], ['ㅏ', 'Eomi']]
    [['지하철', 'Noun']]
    [['기차', 'Noun']]
    [['아침', 'Noun'], ['에', 'Josa'], ['몇', 'Determiner'], ['시', 'Noun'], ['에', 'Josa'], ['타', 'Verb'], ['고', 'Eomi'], ['가', 'Verb'], ['는데', 'Eomi']]

CRF 의 potential function 은 FeatureTransformer class 입니다. 이 classes 은 call 함수가 구현되어 있습니다. (단어, 품사) 로 이뤄진 list 형식의 문장을 입력 받으면, 각 시점의 features 와 tags list 를 return 합니다.

    from crf_postagger import HMMStyleFeatureTransformer
    from crf_postagger import TrigramFeatureTransformer


    # sentence_to_xy = HMMStyleFeatureTransformer()
    sentence_to_xy = TrigramFeatureTransformer()

    sentence = [['뭐', 'Noun'], ['타', 'Verb'], ['고', 'Eomi'], ['가', 'Verb'], ['ㅏ', 'Eomi']]
    features, tags = sentence_to_xy(sentence)

아래는 위의 문장의 features 와 tags 입니다. TrigramFeatureTransformer 은 x[0] / x[0] & y[-1] / x[0:1] / x[0:1] & y[1] / x[-1,1] / x[-1:1] 을 features 로 이용합니다.

    #### features ####
    [['x[0]=뭐',
      'x[0]=뭐, y[-1]=BOS',
      'x[-1:0]=BOS-뭐',
      'x[0:1]=뭐-타',
      'x[0:1]=뭐-타, y[1]=Verb',
      'x[-1,1]=BOS-타',
      'x[-1:1]=BOS-뭐-타'],
     ['x[0]=타',
      'x[0]=타, y[-1]=Noun',
      'x[-1:0]=뭐-타',
      'x[0:1]=타-고',
      'x[0:1]=타-고, y[1]=Eomi',
      'x[-1,1]=뭐-고',
      'x[-1:1]=뭐-타-고'],
     ['x[0]=고',
      'x[0]=고, y[-1]=Verb',
      'x[-1:0]=타-고',
      'x[0:1]=고-가',
      'x[0:1]=고-가, y[1]=Verb',
      'x[-1,1]=타-가',
      'x[-1:1]=타-고-가'],
     ['x[0]=가',
      'x[0]=가, y[-1]=Eomi',
      'x[-1:0]=고-가',
      'x[0:1]=가-ㅏ',
      'x[0:1]=가-ㅏ, y[1]=Eomi',
      'x[-1,1]=고-ㅏ',
      'x[-1:1]=고-가-ㅏ'],
     ['x[0]=ㅏ',
      'x[0]=ㅏ, y[-1]=Verb',
      'x[-1:0]=가-ㅏ',
      'x[0:1]=ㅏ-EOS',
      'x[0:1]=ㅏ-EOS, y[1]=EOS',
      'x[-1,1]=가-EOS',
      'x[-1:1]=가-ㅏ-EOS']]

    #### tags ####
    ('Noun', 'Verb', 'Eomi', 'Verb', 'Eomi')

모델의 학습을 위하여 Trainer 를 이용합니다. Trainer 에는 corpus, feature transformer 를 반드시 입력해야 합니다. max_iter, l1_cost, l2_cost, verbose 는 default 로 설정되어 있습니다. l1_cost = 0 으로 설정하면 L2 regularization 만 적용됩니다. 만약 l2_cost = 0, l1_cost > 0 으로 설정하면 L1 regularized CRF 가 학습됩니다.

    from crf_postagger import Trainer

    trainer = Trainer(
        Corpus(corpus_path, num_sent=-1),
        sentence_to_xy = sentence_to_xy,
        max_iter = 30,
        l1_cost = 0,
        l2_cost = 1,
        verbose = True
    )

학습한 Trainer 를 JSON 형식으로 저장할 수 있습니다.

    model_path = '../models/trigram_crf_sejong_simple.json'
    trainer._save_as_json(model_path)

JSON 파일에는 다음의 정보가 포함되어 있습니다.

    dict_keys(['state_features', 'transitions', 'idx2feature', 'features'])

state_features 은 {feature --> tag : coefficient} 형식의 dict 이며 transitions 은 {'Noun -> Josa': prob} 형식의 dict 입니다. 

### Tagging

학습된 형태소 분석기를 이용하려면 Parameter 와 Tagger 를 import 해야 합니다. Parameter 는 HMMStyleParameter, TrigramParameter 두 종류가 구현되어 있습니다. 각 Parameter 에 따라 class attribute 가 다르게 정의되어 있습니다.

Parameter 와 Tagger 는 반드시 짝을 맞춰서 입력해야 합니다. 이 부분은 이후에 통합될 예정입니다.

    from crf_postagger import TrigramParameter
    from crf_postagger import TrigramTagger

    model_path = '../models/trigram_crf_sejong_simple.json'
    trained_crf = TrigramTagger(
        TrigramParameter(model_path)
    )

Tagger 는 evaluation 기능을 제공합니다. [(단어, 품사), (단어, 품사), ... ] 형식의 문장을 입력하면 score 를 계산합니다.

    candidates = [
        [('뭐', 'Noun'), ('타', 'Verb'), ('고', 'Eomi'), ('가', 'Noun')],
        [('뭐', 'Noun'), ('타', 'Verb'), ('고', 'Noun'), ('가', 'Noun')],
        [('뭐', 'Noun'), ('타', 'Verb'), ('고', 'Eomi'), ('가', 'Verb'), ('ㅏ', 'Eomi')],
        [('뭐', 'Noun'), ('타', 'Verb'), ('고', 'Noun'), ('가', 'Verb'), ('ㅏ', 'Eomi')]
    ]

    for sent in candidates:
        print('\n{}'.format(sent))
        print(trained_crf.score(sent, debug=False))

    [('뭐', 'Noun'), ('타', 'Verb'), ('고', 'Eomi'), ('가', 'Noun')]
    27.858939

    [('뭐', 'Noun'), ('타', 'Verb'), ('고', 'Noun'), ('가', 'Noun')]
    -3.6017170000000003

    [('뭐', 'Noun'), ('타', 'Verb'), ('고', 'Eomi'), ('가', 'Verb'), ('ㅏ', 'Eomi')]
    54.225075

    [('뭐', 'Noun'), ('타', 'Verb'), ('고', 'Noun'), ('가', 'Verb'), ('ㅏ', 'Eomi')]
    23.032051000000003

tag 함수는 str 형식의 문장에 대하여 형태소 분석 결과를 return 합니다.  k 는 beam-search 의 beam 크기 입니다. return 되는 분석 결과도 k 개 입니다.

    sent = '주간아이돌에서아이오아이가소개했던오늘의날씨'

    paths = trained_crf.tag(sent, k=3)
    for path, score in paths:
        print('\nscore = {}'.format(score))
        for pos in path:
            print(pos)

학습에 이용한 세종 말뭉치에는 '주간아이돌', '아이오아이'라는 단어가 존재하지 않기 때문에 이를 분해하였습니다.

    score = 140.8013020000001
    ('주간', 'Noun', 0, 2)
    ('아이돌', 'Noun', 2, 5)
    ('에', 'Josa', 5, 6)
    ('서', 'Verb', 6, 7)
    ('아', 'Eomi', 7, 8)
    ('이오', 'Noun', 8, 10)
    ('아이', 'Noun', 10, 12)
    ('가', 'Josa', 12, 13)
    ('소개', 'Noun', 13, 15)
    ('하 + 았', 'Verb + Eomi', 15, 16)
    ('던', 'Eomi', 16, 17)
    ('오늘', 'Noun', 17, 19)
    ('의', 'Josa', 19, 20)
    ('날씨', 'Noun', 20, 22)

    score = 140.8013020000001
    ('주간', 'Noun', 0, 2)
    ('아이돌', 'Noun', 2, 5)
    ('에', 'Josa', 5, 6)
    ('서', 'Verb', 6, 7)
    ('아', 'Eomi', 7, 8)
    ('이오', 'Noun', 8, 10)
    ('아이', 'Noun', 10, 12)
    ('가', 'Josa', 12, 13)
    ('소개', 'Noun', 13, 15)
    ('하 + 았', 'Adjective + Eomi', 15, 16)
    ('던', 'Eomi', 16, 17)
    ('오늘', 'Noun', 17, 19)
    ('의', 'Josa', 19, 20)
    ('날씨', 'Noun', 20, 22)

    score = 137.47260700000007
    ('주간', 'Noun', 0, 2)
    ('아이돌', 'Noun', 2, 5)
    ('에', 'Josa', 5, 6)
    ('서', 'Verb', 6, 7)
    ('아', 'Eomi', 7, 8)
    ('이오', 'Noun', 8, 10)
    ('아이', 'Noun', 10, 12)
    ('가', 'Josa', 12, 13)
    ('소개', 'Noun', 13, 15)
    ('했던', 'Unk', 15, 17)
    ('오늘', 'Noun', 17, 19)
    ('의', 'Josa', 19, 20)
    ('날씨', 'Noun', 20, 22)
    
이 구현체는 사용자 사전을 추가할 수 있는 기능을 제공합니다. add_user_dictionary 에는 (tag, {word: preference}) 를 입력합니다. 입력되는 단어 word 가 tag 일 점수가 preference 만큼 더해집니다.

    trained_crf.add_user_dictionary('Noun', {'아이오아이':20, '주간아이돌':30})

    paths = trained_crf.tag(sent, k=1)
    for path, score in paths:
        print('\nscore = {}'.format(score))
        for pos in path:
            print(pos)


이번에는 편의를 위하여 k=1 로 tag 를 수행하였습니다. '주간아이돌'과 '아이오아이'가 제대로 인식됩니다.

    score = 161.9815650000001
    ('주간아이돌', 'Noun', 0, 5)
    ('에서', 'Josa', 5, 7)
    ('아이오아이', 'Noun', 7, 12)
    ('가', 'Josa', 12, 13)
    ('소개', 'Noun', 13, 15)
    ('하 + 았', 'Verb + Eomi', 15, 16)
    ('던', 'Eomi', 16, 17)
    ('오늘', 'Noun', 17, 19)
    ('의', 'Josa', 19, 20)
    ('날씨', 'Noun', 20, 22)

[crf_tagger_post]: https://lovit.github.io/nlp/2018/09/13/crf_based_tagger/