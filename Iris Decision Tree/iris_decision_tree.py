import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import tree

data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
#df.head(10)

X_train, X_test, Y_train, Y_test = train_test_split(df[data.feature_names], df['target'], random_state=0)

clf = DecisionTreeClassifier(max_depth = 2, random_state = 0)

clf.fit(X_train, Y_train)

clf.predict(X_test)

score = clf.score(X_test, Y_test)
print(score)

fig, axes = plt.subplots(nrows = 1, ncols = 1, figsize = (4,4), dpi = 200)

tree.plot_tree(clf); #this line alone is enough
#fig.savefig('iris_decision_tree(not_labelled).png')

fn = ['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']
cn = ['setosa', 'versicolor', 'virginica']

fig, axes = plt.subplots(nrows = 1, ncols = 1, figsize = (4,4), dpi = 200)

tree.plot_tree(clf, feature_names = fn, class_names=cn, filled = True);
#fig.savefig('iris_decision_tree(labelled).png')

plt.show()

#tree.export_graphviz(clf, out_file="iris_decision_tree.dot", feature_names = fn, class_names=cn, filled = True)
