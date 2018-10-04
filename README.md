## Conditional Random Field (CRF) 기반 한국어 형태소 분석기

세종 말뭉치를 이용하여 학습한 CRF 기반 한국어 형태소 분석기입니다. CRF 을 이용하여 형태소 분석을 하는 과정을 설명하기 위한 코드입니다.

학습을 위해서는 crfsuite 의 Python wrapper 인 python-crfsuite 를 이용하였습니다. Decoding 과 tagging 과정은 Python 코드로만 이뤄져 있습니다. python-crfsuite 의 tagging 기능을 이용하지 않습니다.

이 repository 에서는 CRF 모델을 학습하는 Trainer 와, 학습된 모델을 이용하여 품사 판별을 하는 HMMStyleTagger 및 TrigramTagger 을 제공합니다.

구현 과정 및 원리에 대해서는 [블로그 포스트][crf_tagger_post]를 참고하세요.

## Usage

### Tagging

학습된 형태소 분석기를 이용하려면 Parameter 와 Tagger 를 import 해야 합니다. Parameter 는 HMMStyleParameter, TrigramParameter 두 종류가 구현되어 있습니다. 각 Parameter 에 따라 class attribute 가 다르게 정의되어 있습니다.

Parameter 와 Tagger 는 반드시 짝을 맞춰서 입력해야 합니다. 이 부분은 이후에 통합될 예정입니다. 용언에 대한 기분석 어절의 결과를 입력할 수 있습니다. { '어절':((어간, 어미, 어간 품사, 어미 품사), ) } 형식으로 입력합니다.

    from crf_postagger.trigram import TrigramTagger
    from crf_postagger.trigram import TrigramParameter

    model_path = '../models/trigram_crf_sejong_simple.json'
    preanalyzed_eojeols = {
        '해쪄': (('하', '아쪄', 'Verb', 'Eomi'),)
    }

    trained_crf = TrigramTagger(
        TrigramParameter(
            model_path,
            preanalyzed_eojeols = preanalyzed_eojeols
        )
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
        print(trained_crf.evaluate(sent, debug=False))

정답인 [('뭐', 'Noun'), ('타', 'Verb'), ('고', 'Eomi'), ('가', 'Verb'), ('ㅏ', 'Eomi')] 가 가장 큰 점수를 받았습니다.

    [('뭐', 'Noun'), ('타', 'Verb'), ('고', 'Eomi'), ('가', 'Noun')]
    2.037970832351861

    [('뭐', 'Noun'), ('타', 'Verb'), ('고', 'Noun'), ('가', 'Noun')]
    -0.26347716229917634

    [('뭐', 'Noun'), ('타', 'Verb'), ('고', 'Eomi'), ('가', 'Verb'), ('ㅏ', 'Eomi')]
    3.9667383324286725

    [('뭐', 'Noun'), ('타', 'Verb'), ('고', 'Noun'), ('가', 'Verb'), ('ㅏ', 'Eomi')]
    1.684868477842624

학습 데이터에 존재하지 않은 단어를 다음과 같이 입력할 수 있습니다. x[0] 형식의 features 로 이용됩니다. 사용자 사전은 preference 의 역할도 수행합니다. ('품사', {'단어': preference}) 형태로 add_user_dictionary 에 선호하는 단어를 입력합니다.

Coefficients 는 [-1, 1] 사이로 scaling 이 되어 있습니다. 다른 어떤 features 보다도 영향력을 크게 만드려면 1 보다 큰 값을 입력해도 됩니다. CRF 가 features 의 coefficients 를 학습할 때 자주 등장하지 않은 단어의 x[0] 의 coefficient 를 작게 학습하는 경향이 있습니다. 이는 user preference 로 조절할 수 있습니다.

    trained_crf.add_user_dictionary('Noun', {'아이오아이':1, '아이돌룸':1})
    trained_crf.add_user_dictionary('Verb', {'나오':0.5})
    sent = '아이돌룸에아이오아이가나올수있을까'

    trained_crf._a_syllable_penalty = -0.3
    trained_crf._noun_preference = 0.2

    top_poses = trained_crf.tag(sent, flatten=True)
    for poses, score in top_poses:
        print('\nscore = {}'.format(score))
        for pos in poses:
            print(pos)

flatten=True 가 기본값이며, 주어진 문장에 대해 가장 적절한 (형태소열, 점수) 형태로 return 됩니다.

    score = 8.741956263158434
    ('아이돌룸', 'Noun')
    ('에', 'Josa')
    ('아이오아이', 'Noun')
    ('가', 'Josa')
    ('나오', 'Verb')
    ('ㄹ', 'Eomi')
    ('수', 'Noun')
    ('있', 'Verb')
    ('을까', 'Eomi')

    score = 8.700471735794014
    ('아이돌룸', 'Noun')
    ('에', 'Josa')
    ('아이오아이', 'Noun')
    ('가', 'Josa')
    ('나오', 'Verb')
    ('ㄹ', 'Eomi')
    ('수', 'Noun')
    ('있', 'Verb')
    ('을', 'Eomi')
    ('끄', 'Verb')
    ('아', 'Eomi')

    score = 8.631769824885843
    ('아이돌룸', 'Noun')
    ('에', 'Josa')
    ('아이오아이', 'Noun')
    ('가', 'Josa')
    ('나오', 'Verb')
    ('ㄹ', 'Eomi')
    ('수', 'Noun')
    ('있', 'Verb')
    ('을', 'Eomi')
    ('까', 'Eomi')

    score = 8.506254643400048
    ('아이돌룸', 'Noun')
    ('에', 'Josa')
    ('아이오아이', 'Noun')
    ('가나', 'Noun')
    ('오', 'Verb')
    ('ㄹ', 'Eomi')
    ('수', 'Noun')
    ('있', 'Verb')
    ('을', 'Eomi')
    ('끄', 'Verb')
    ('아', 'Eomi')

    score = 8.437552732491877
    ('아이돌룸', 'Noun')
    ('에', 'Josa')
    ('아이오아이', 'Noun')
    ('가나', 'Noun')
    ('오', 'Verb')
    ('ㄹ', 'Eomi')
    ('수', 'Noun')
    ('있', 'Verb')
    ('을', 'Eomi')
    ('까', 'Eomi')

flatten=False 로 설정하면 형태소의 위치와 단어 점수가 함께 출력됩니다. 또한 beam_size 를 설정하면 해당 개수 만큼의 후보가 return 됩니다.

    poses, score = trained_crf.tag(sent, flatten=False, beam_size=1)[0]
    print(score)
    for pos in poses:
        print('score = {}'.format(score))

    score = 8.700471735794014
    ('아이돌룸/Noun', 0, 4, 1)
    ('에/Josa', 4, 5, 0.8190128120533081)
    ('아이오아이/Noun', 5, 10, 1)
    ('가/Josa', 10, 11, 0.41837169731542345)
    ('나오/Verb + ㄹ/Eomi', 11, 13, 0.6485539073324389)
    ('수/Noun', 13, 14, 0.18038937990949483)
    ('있/Verb', 14, 15, 0.4063281916380028)
    ('을/Eomi', 15, 16, 0.3606708588333233)
    ('끄/Verb + 아/Eomi', 16, 17, 0.42202906523364403)

### Tagging HMM-style CRF tagger

용언에 대하여 기분석 어절을 이용할 수 있습니다. Tagger 는 학습된 모델인 Parameter 를 입력해야 합니다. 이는 이후에 통합될 예정입니다.

    from crf_postagger.hmm_style import HMMStyleTagger
    from crf_postagger.hmm_style import HMMStyleParameter

    model_path = '../models/hmmstyle_crf_sejong_simple.json'
    preanalyzed_lemmas = {
        '해쪄': (('하', '아쪄', 'Verb', 'Eomi'),)
    }

    trained_crf = HMMStyleTagger(
        HMMStyleParameter(
            model_path,
            preanalyzed_lemmas = preanalyzed_lemmas
        )
    )

임의의 단어 품사 열에 대한 evaluation 이 가능합니다. 단어 품사 열은 (단어, 품사) 형태의 list 입니다.

    sentence = [['뭐', 'Noun'], ['타', 'Verb'], ['고', 'Eomi'], ['가', 'Verb'], ['ㅏ', 'Eomi']]
    print(trained_crf.evaluate(sentence)) # 2.728246174403568

str 형식의 문장에 대한 형태소 분석은 tag 함수를 이용합니다. tag 는 단어열과 단어열의 점수가 함께 return 됩니다.

    print(trained_crf.tag('머리쿵해쪄'))

    [[('머리', 'Noun'),
      ('쿵', 'Adverb'),
      ('하', 'Verb'),
      ('아', 'Eomi'),
      ('찌', 'Verb'),
      ('어', 'Eomi')],
     2.0720928527608016]

'해쪄' 와 같은 대화체는 세종말뭉치에 존재하지 않기 때문에 제대로 인식이 되지 않습니다. 사용자 사전은 preference 의 역할도 수행합니다. ('품사', {'단어': preference}) 형태로 add_user_dictionary 에 선호하는 단어를 입력합니다.

    trained_crf.add_user_dictionary('Eomi', {'아쪄':1, '아써':1})
    print(trained_crf.tag('머리쿵해쪄'))

    [[('머리', 'Noun'), ('쿵', 'Adverb'), ('하', 'Verb'), ('아쪄', 'Eomi')],
     2.797982565508236]

tag(flatten = False) 로 설정하면 형태소의 위치와 단어 점수가 함께 출력됩니다.

    print(trained_crf.tag('머리쿵해쪄', flatten=False))

    [[('머리/Noun', 0, 2, 0.029517693554896852),
      ('쿵/Adverb', 2, 3, 0.0021641369892205606),
      ('하/Verb + 아쪄/Eomi', 3, 5, 1.6799258724759993)],
     2.797982565508236]

### Training (HMM-style, Trigram both)

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

    from crf_postagger.hmm_style import HMMStyleFeatureTransformer
    from crf_postagger.trigram import TrigramFeatureTransformer


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

## TODO

unknown word 에 대한 inference 기능은 구현하지 않았습니다.

[crf_tagger_post]: https://lovit.github.io/nlp/2018/09/13/crf_based_tagger/