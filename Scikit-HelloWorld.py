import pandas
from sklearn.metrics import accuracy_score
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

#leitura do dataset Iris
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

#divisão entre features e target do dataset iris
array = dataset.values
X = array[:,0:4]
Y = array[:,4]

#divisão do conjunto de treino e teste considerando 80% para treino e 20% para teste
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.2, random_state=7)


#criando uma lista com vários modelos
models = []
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))


#rodando cada um dos modelos e imprimindo a acurácia
for name, model in models:
    model = model.fit(X_train, Y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(pred, Y_test)
    print(name + ' ' + "{0:.2f}".format(acc)+'%')