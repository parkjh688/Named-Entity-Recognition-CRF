import os
import pycrfsuite
from konlpy.tag import Twitter


def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    features = [
        'bias',
        'word.lower=' + word.lower(),
        'word[-3:]=' + word[-3:],
        'word[-2:]=' + word[-2:],
        'word.isupper=%s' % word.isupper(),
        'word.istitle=%s' % word.istitle(),
        'word.isdigit=%s' % word.isdigit(),
        'postag=' + postag,
        'postag[:2]=' + postag[:2],
    ]
    if i > 0:
        word1 = sent[i - 1][0]
        postag1 = sent[i - 1][1]
        features.extend([
            '-1:word.lower=' + word1.lower(),
            '-1:word.istitle=%s' % word1.istitle(),
            '-1:word.isupper=%s' % word1.isupper(),
            '-1:postag=' + postag1,
            '-1:postag[:2]=' + postag1[:2],
        ])
    else:
        features.append('BOS')

    if i < len(sent) - 1:
        word1 = sent[i + 1][0]
        postag1 = sent[i + 1][1]
        features.extend([
            '+1:word.lower=' + word1.lower(),
            '+1:word.istitle=%s' % word1.istitle(),
            '+1:word.isupper=%s' % word1.isupper(),
            '+1:postag=' + postag1,
            '+1:postag[:2]=' + postag1[:2],
        ])
    else:
        features.append('EOS')

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [label for token, postag, label in sent]


def is_filepath_existed(filepath='model/ner.crf'):
    filepath = filepath.split('/')

    file = filepath[-1]
    filepath.remove(file)

    directory = '/'.join(filepath)

    if not os.path.exists(directory):
        os.makedirs(directory)


def train(filepath='', x_data=[], y_data=[]):
    x_train = []
    y_train = []
    twitter = Twitter()

    # make train data
    for text, label in zip(x_data, y_data):
        sent = twitter.pos(text, norm=True, stem=True)
        sent = [(sent[i][0], sent[i][1], label[i]) for i in range(len(sent))]
        x_train.append(sent2features(sent))
        y_train.append(sent2labels(sent))

    # make file path directory
    is_filepath_existed(filepath)

    # train with pycrfsuite
    crf_trainer = pycrfsuite.Trainer(verbose=False)
    for x_seq, y_seq in zip(x_train, y_train):
        crf_trainer.append(x_seq, y_seq)

    print('crf train start!')
    crf_trainer.train(filepath)

    return True


def predict(filepath='', x_test=[]):
    twitter = Twitter()
    predicted_labels = []
    # file check
    is_filepath_existed(filepath)

    tagger = pycrfsuite.Tagger()
    tagger.open(filepath)

    for text in x_test:
        sent = twitter.pos(text, norm=True, stem=True)
        sent = [(sent[i][0], sent[i][1]) for i in range(len(sent))]
        predicted_labels.append(tagger.tag(sent2features(sent)))

    return predicted_labels
