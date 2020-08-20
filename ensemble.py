import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

bert = np.load('BERT_preds.npy')                #covid-twitter
bertProbs = np.load('BERT_preds_proba.npy')     #covid-twitter
svm = np.load('SVM_preds.npy')
conv = np.load('CONV_preds.npy')
convProbs = np.load('CONV_preds_proba.npy')
mlp = np.load('MLP_preds.npy')
mlpProbs = np.load('MLP_preds_proba.npy')
##############
baseBert = np.load("BERT_base_preds.npy")
baseBertProbs = np.load("BERT_base_preds_proba.npy")
roberta = np.load("RoBERTa_preds.npy")
robertaProbs = np.load("RoBERTa_preds_proba.npy")


golds = [line.strip().split('\t')[-1] =='INFORMATIVE' for line in open('data/valid.tsv').readlines()][1:]

feats = []
for i in range(len(bert)):
    instance = [bert[i], bertProbs[i][0], bertProbs[i][1], svm[i], conv[i], convProbs[i], mlp[i], mlpProbs[i], baseBert[i], baseBertProbs[i][0],baseBertProbs[i][1], roberta[i], robertaProbs[i][0],robertaProbs[i][1]]
    for j in range(len(instance)):
        if type(instance[j]) == np.ndarray:
            instance[j] = float(instance[j][0])
        else:
            instance[j] = float(instance[j])
    feats.append(instance)

preds = []
numFolds = 10
for fold in range(numFolds):
    devBeg = int((len(bert) /numFolds) * fold)
    devEnd = int((len(bert) /numFolds) * (fold + 1))
    trainGold = golds[0:devBeg] + golds[devEnd:]
    trainFeats = feats[0:devBeg] + feats[devEnd:]
    devFeats = feats[devBeg:devEnd]

    #classifier = LinearSVC()
    #classifier = LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=1.0, multi_class='ovr', fit_intercept=True, intercept_scaling=1, class_weight=None)
    classifier = RandomForestClassifier(max_depth=5, random_state=0, n_estimators=200)
    #classifier = LogisticRegression()
    classifier.fit(trainFeats, trainGold)
    preds.extend(classifier.predict(devFeats))

#for i in range(len(feats)):
#    preds = [bert[i], svm[i], conv[i], mlp[i]]
#    cors = 0
#    for pred in preds:
#        if pred == golds[i]:
#            cors += 1
#    print(cors)

cor = 0
for gold, pred in zip(golds, preds):
    if gold == pred:
        cor += 1
print(cor/len(bert))


