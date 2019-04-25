import codecs
import os
import re

import jieba
import numpy as np


# 加载数据
def load_files(to_path, filename):
    corpus = []
    for files in os.listdir(filename):
        curr_path = os.path.join(filename, files)
        if os.path.isdir(curr_path):
            count = 0
            docs = []
            for file in os.listdir(curr_path):
                count = count + 1
                file_path = os.path.join(curr_path, file)
                with codecs.open(file_path, 'r', encoding='utf-8') as f:
                    docs.append(files + '\t' + f.read())
        corpus.append(docs)
    with codecs.open(to_path, 'w', encoding='utf-8') as fp:
        for docs in corpus:
            for doc in docs:
                fp.write(doc + '\n')
    return corpus


# 保存标签和文本
def split_data_with_label(corpus):
    input_x = []
    input_y = []
    stopwords = []
    with codecs.open('./spamDataSet/english') as f:
        stopwords.append(f.read())
    re_han = re.compile(u"([\u4E00-\u9FD5a-zA-Z0-9+#&\._%]+)")  # the method of cutting text by punctuation
    if os.path.isfile(corpus):
        with codecs.open(corpus, 'r', encoding='utf=8') as f:
            for line in f:
                try:
                    line = line.rstrip()
                    assert len(line.split('\t')) == 2
                    label, content = line.split('\t')
                    blocks = re_han.split(content)
                    word = []
                    for blk in blocks:
                        if re_han.match(blk):
                            word.extend(jieba.lcut(blk))
                            word = [x.lower() for x in word if x not in stopwords]
                    input_x.append(word)
                    if label == 'neg':
                        input_y.append(0)
                    else:
                        input_y.append(1)

                except:
                    pass
    return input_x, input_y


# 词袋模型
def createVocabList(input_x):
    vocabSet = set([])
    for document in input_x:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


# 统计词语，存在则为1
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
    return returnVec


def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            print(1)
            p1Denom += sum(trainMatrix[i])
        else:
            print(0)
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = np.log(p1Num / p1Denom)
    p0Vect = np.log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive


# 朴素贝叶斯分类函数
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


if __name__ == '__main__':
    train_path = './spamDataset/email/train'
    test_path = './spamDataset/email/test'
    datalist = load_files('./spamDataset/all_text', train_path)
    testlist = load_files('./spamDataset/test_text', test_path)
    input_x, input_y = split_data_with_label('./spamDataset/all_text')
    test_x, test_y = split_data_with_label('./spamDataset/test_text')
    myVocablist = createVocabList(input_x)
    trainMat = []
    print(myVocablist)
    print(input_x)
    for postinDoc in input_x:
        trainMat.append(setOfWords2Vec(myVocablist, postinDoc))
    print(trainMat)
    p0V, p1V, pA = trainNB0(trainMat, input_y)
    print(p0V)

    print(p1V)
    result = []
    for testEntry in test_x:
        print(testEntry)
        thisDoc = setOfWords2Vec(myVocablist, testEntry)
        print(thisDoc)
        result.append(classifyNB(thisDoc, p0V, p1V, pA))
    print(result)
    print(test_y)
