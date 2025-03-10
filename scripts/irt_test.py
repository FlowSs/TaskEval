import numpy as np
import os
import sys
from sklearn.datasets import make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis,LinearDiscriminantAnalysis
from sklearn.metrics import *
import matplotlib.pyplot as plt
import seaborn as sns

def synth_data(size):
    size = int(size/2)
    c1 = np.random.multivariate_normal([1, 1], np.diag([1, 1]), size=size)
    c2 = np.random.multivariate_normal([3, 3], np.diag([1, 1]), size=size)
    X = np.vstack((c2, c1))
    y = np.zeros(len(X))
    y[-size:] = 1

    return X, y


def positive_baseline(y_test):
    probas = np.ones(len(y_test)).reshape(-1, 1)
    predictions = np.ones(len(y_test))
    probas = np.hstack((1.0 - probas, probas))
    return probas, predictions.astype(int)

def negative_baseline(y_test):
    probas = np.zeros(len(y_test)).reshape(-1, 1)
    predictions = np.zeros(len(y_test))
    probas = np.hstack((1.0 - probas, probas))
    return probas, predictions.astype(int)


def uncertain_baseline(y_test):
    probas = 0.5 * np.ones(len(y_test)).reshape(-1, 1)
    #probas = np.random.normal(loc=0.5,scale=0.1,size=len(y_test)).reshape(-1,1)
    probas = np.hstack((1.0 - probas, probas))
    #predictions = np.ones(len(y_test))
    #predictions = np.argmax(probas,axis=1)
    predictions = np.random.choice(2,len(y_test))
    #print(predictions.shape)
    return probas, predictions.astype(int)


def get_probas_and_predictions(classifier, X_test, y_test, pclassifier=None):
    if classifier == 'weak':
        return weak_baseline(y_test)
    if classifier == 'strong':
        return strong_baseline(y_test)
    if classifier == 'positive':
        return positive_baseline(y_test)
    if classifier == 'negative':
        return negative_baseline(y_test)
    if classifier == 'uncertain':
        return uncertain_baseline(y_test)
    if isinstance(classifier,str) and 'perturb' in classifier:
        print(classifier)
        probas = pclassifier.predict_proba(X_test)
        #predictions = pclassifier.predict(X_test).astype(int)
        n = len(y_test)
        pidx = np.random.choice(n, int(0.4 * n), replace=False)
        probas[pidx,0] = np.random.uniform(0.,1.,size=len(pidx))
        probas[pidx,1] = 1 - probas[pidx,0]
        predictions = np.argmax(probas,axis=1)
        #print(probas.shape,predictions.shape)
        return probas,predictions
        
    probas = classifier.predict_proba(X_test)
    predictions = classifier.predict(X_test).astype(int)

    return probas, predictions


def evaluate(classifier, X_test, y_test,recal,pclassifier=None):
    if isinstance(classifier,str) and 'perturb' in classifier:
        probas, predictions = get_probas_and_predictions(classifier, X_test, y_test, pclassifier)
    else:
        probas, predictions = get_probas_and_predictions(classifier, X_test, y_test)
    incorrect = predictions != y_test
    correct = predictions == y_test

    if recal:
        lr = LogisticRegression()
        lr.fit(probas,predictions)
        probas = lr.predict_proba(probas)
 

    chance = probas[np.arange(len(probas)), predictions]
    chance[incorrect] = 1.0 - chance[incorrect]

    return probas[:, 1], chance, np.array([[np.sum(correct) + 1, np.sum(
     incorrect) + 1]]), correct, incorrect

def cl_info(classifier, X_train, y_train, X_test, y_test,recal=False):
    pclassifier = None
    if not isinstance(classifier,str):
        classifier.fit(X_train, y_train)
    elif 'perturb' in classifier:
        pclassifier = classifiers[classifier.split('-')[1]]
        pclassifier.fit(X_train, y_train)
    p, p_irt, prior, correct, incorrect = evaluate(classifier, X_test, y_test,recal,pclassifier)
    return p, p_irt, prior, correct, incorrect

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'birt-gd', 'src'))

from birt import Beta3


if __name__ == '__main__':
   
   classifiers = {
      'MLP': MLPClassifier(max_iter=1000,hidden_layer_sizes=(256,64,)),##
      'Logistic Regression': LogisticRegression(),##
      'Random Forest': RandomForestClassifier(),
      'Nearest Neighbors': KNeighborsClassifier(3),##
      'Adaboost': AdaBoostClassifier(),
      'Naive Bayes': GaussianNB(),#
      'LDA': LinearDiscriminantAnalysis(),
      'QDA': QuadraticDiscriminantAnalysis(),
      'Positive Classifier': 'positive',
      'Negative Classifier': 'negative',
      'Constant Classifier': 'uncertain',##
   }

   X, y = synth_data(800) #make_circles(n_samples=800)
   # scaler = StandardScaler()
   # scaler.fit(X)
   # X = scaler.transform(X)

   
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 42)

   noise_frac = 0.2
   n = len(y_test)
   nidx = np.random.choice(n, int(noise_frac * n), replace=False)
   y_test[nidx] = 1 - y_test[nidx]

   for i in range(2):
      plt.scatter(X_test[y_test == i, 0], X_test[y_test == i, 1])
   plt.show()

   correct = np.ones(len(y_test))
   incorrect = np.ones(len(y_test))
   probas = []
   probas_irt = []
   cl_priors = []
   labels = []
   accuracy = []
   f1 = []
   Waccuracy = []

   for classifier in classifiers:
         labels.append(classifier)
         cl = classifiers[classifier]
         p, p_irt, prior, c, i = cl_info(cl, X_train, y_train, X_test, y_test,recal=False)
         pred = np.zeros_like(c)
                  
         Waccuracy.append(c.sum()*1./len(y_test))  
         pred[c] = y_test[c]
         pred[i] = 1 - y_test[i]
         f1.append(f1_score(y_test,pred, average=None))
         correct += c
         incorrect += i
         probas.append(p)
         
         # p_list = []
         
         # for p in p_irt:
         #       if p == 1.0:
         #          p_list.append(p - 1e-3)
         #       elif p == 0.0:
         #          p_list.append(p + 1e-3)
         #       else:
         #          p_list.append(p)

         probas_irt.append(p_irt)
         cl_priors.append(prior)

   print(list(classifiers.keys()))
   print(f1)

   X_test = np.array(X_test)
   y_test = np.array(y_test)
   probas_irt = np.array(probas_irt).T

   b3 = Beta3(n_respondents=probas_irt.shape[1], n_items=probas_irt.shape[0], epochs=3000, n_inits=300, random_seed=0, learning_rate=1, n_workers=-1, tol=1e-7)
   b3.fit(probas_irt)

   scatter = plt.scatter(X_test[:,0], X_test[:,1], c=np.array(b3.difficulties), cmap = sns.cubehelix_palette(rot=-.5,light=1.5,dark=-.5,as_cmap=True), edgecolor='black')
   plt.colorbar(scatter, label='Value')
   plt.title('Difficulty of instances')
   plt.show()
   scatter = plt.scatter(X_test[:,0], X_test[:,1], c=np.array(b3.discriminations), cmap = sns.cubehelix_palette(rot=-.5,light=1.5,dark=-.5,as_cmap=True), edgecolor='black')
   plt.colorbar(scatter, label='Value')
   plt.title('Discriminant of instances')
   plt.show()
