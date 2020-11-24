from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.mixture import GaussianMixture
from sklearn import tree
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, roc_auc_score 
import numpy as np
import pandas as pd
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

def classifier_main(X_trian, y_train, X_test, y_test, classifier = 'log'):
    if classifier == 'svm':
        clf = svm.SVC(probability=True, kernel = 'rbf').fit(X_trian,y_train)
    elif classifier == 'log':
        clf = LogisticRegression(random_state=0).fit(X_trian, y_train)
    elif classifier == 'tree':
        clf = tree.DecisionTreeClassifier(max_depth = 2).fit(X_trian, y_train)
        # tree.plot_tree(clf)
        # plt.title('Decision Tree, Vowel: {vowel}')
        # plt.show()
    else:
        print('Wrong input - defaulting to logistic regression')
        clf = LogisticRegression(random_state=0).fit(X_trian, y_train)
    
    print(f".::| CLASSIFIER: {classifier} |::.")
    print('TRAIN ROC AUC: {:.3f}'.format(roc_auc_score(y_train, clf.predict_proba(X_trian)[:,1])))
    print('TRAIN Accuracy: {:.3f}'.format(accuracy_score(y_train, clf.predict(X_trian))))
    print('TEST ROC AUC: {:.3f}'.format(roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])))
    print('TEST Accuracy: {:.3f}'.format(accuracy_score(y_test, clf.predict(X_test))))

def classifier(X_a, y_a, vowel_a, X_i, y_i, classifier = 'log', vowel = 'a'):
    """
    classifier = log - logistic regression
               = svm - SVM
    """
    if classifier == 'svm':
        clf = svm.SVC(probability=True, kernel = 'rbf').fit(X_a,y_a)
    elif classifier == 'log':
        clf = LogisticRegression(random_state=0).fit(X_a, y_a)
    elif classifier == 'tree':
        clf = tree.DecisionTreeClassifier(max_depth = 2).fit(X_a, y_a)
        # tree.plot_tree(clf)
        # plt.title('Decision Tree, Vowel: {vowel}')
        # plt.show()
    else:
        print('Wrong input - defaulting to logistic regression')
        clf = LogisticRegression(random_state=0).fit(X_a, y_a)

    print(f".::| CLASSIFIER: {classifier} | VOWEL: {vowel} |::.")
    print('TRAIN ROC AUC: {:.3f}'.format(roc_auc_score(y_a, clf.predict_proba(X_a)[:,1])))
    print('TRAIN Accuracy: {:.3f}'.format(accuracy_score(y_a, clf.predict(X_a))))
    print('TEST ROC AUC: {:.3f}'.format(roc_auc_score(y_i, clf.predict_proba(X_i)[:,1])))
    print('TEST Accuracy: {:.3f}'.format(accuracy_score(y_i, clf.predict(X_i))))


    # fig = plt.figure(figsize=plt.figaspect(0.5))
    # ax = fig.add_subplot(1, 1, 1, projection='3d')
    # ax.scatter3D(vowel_a[vowel_a['is_covid'] == 0]['alpha'], vowel_a[vowel_a['is_covid'] == 0]['beta'], vowel_a[vowel_a['is_covid'] == 0]['delta'], c='b', marker='o', label = 'normal')
    # ax.scatter3D(vowel_a[vowel_a['is_covid'] == 1]['alpha'], vowel_a[vowel_a['is_covid'] == 1]['beta'], vowel_a[vowel_a['is_covid'] == 1]['delta'], c='r', marker='o', label = 'creaky')
    # ax.set_xlabel('alpha')
    # ax.set_ylabel('beta')
    # ax.set_zlabel('delta')
    # plt.title('3D data plot')
    # plt.legend(loc='upper left')
    # plt.show()

if __name__ == '__main__':
    data = pd.read_csv('output/exp_covid_vowel_section_1_with_melplots/results.csv')
    
    # vowel_a_covid = data[(data['# file_name'].str.contains("vowel_a")) & (data['is_covid'] == 1)]
    # vowel_a_normal = data[(data['# file_name'].str.contains("vowel_a")) & (data['is_covid'] == 0)]
    # vowel_i_covid = data[(data['# file_name'].str.contains("vowel_i")) & (data['is_covid'] == 1)]
    # vowel_i_normal = data[(data['# file_name'].str.contains("vowel_i")) & (data['is_covid'] == 0)]

    vowel_a = data[data['# file_name'].str.contains("vowel_a")]
    vowel_i = data[data['# file_name'].str.contains("vowel_i")]
    vowel_i = data[data['# file_name'].str.contains("vowel_i")]

    X_a = vowel_a[['alpha', 'beta', 'delta']].to_numpy()
    X_i = vowel_i[['alpha', 'beta', 'delta']].to_numpy()
    X_u = vowel_i[['alpha', 'beta', 'delta']].to_numpy()
    y_a = vowel_a['is_covid'].to_numpy()
    y_i = vowel_i['is_covid'].to_numpy()
    y_u = vowel_i['is_covid'].to_numpy()

    print('Data size of vowel a: ', X_a.shape[0])
    print('No of covid voices in vowel a: ', X_a[y_a == 1].shape[0])
    print('Data size of vowel i: ', X_i.shape[0])
    print('No of covid voices in vowel i: ', X_i[y_i == 1].shape[0])
    print('Data size of vowel i: ', X_u.shape[0])
    print('No of covid voices in vowel i: ', X_u[y_u == 1].shape[0])

    # clf = LogisticRegression(random_state=0).fit(X, y)
    X = np.concatenate([X_a, X_i, X_u], axis = 0)
    y = np.concatenate([y_a, y_i, y_u], axis = 0)
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.3, random_state=111)

    for c in ['log', 'svm', 'tree']:
        classifier_main(X_train, X_test, y_train, y_test, c)
        # classifier(X_a, y_a, vowel_a, X_i, y_i, c, 'a')
        # classifier(X_i, y_i, vowel_i, X_a, y_a, c, 'i')

    # TWO SUBPLOTS
    # fig = plt.figure(figsize=plt.figaspect(0.5))
    # ax = fig.add_subplot(1, 2, 1, projection='3d')
    # n = ax.scatter3D(OW2[:, 0], OW2[:, 1], OW2[:, 2], c='b', marker='o')
    # ax.title.set_text('Normal OW2 Phoneme')

    # ax = fig.add_subplot(1, 2, 2, projection='3d')
    # c = ax.scatter3D(OW2_c[:, 0], OW2_c[:, 1], OW2_c[:, 2], c='k', marker='o')
    # ax.title.set_text('Creaky OW2 Phoneme')

    # plt.show()

