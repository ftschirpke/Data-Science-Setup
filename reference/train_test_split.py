from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold

"""
You want to split your data into separate training and test sets, and you need to do so randomly.
The scikit-learn library has a function that shuffles and splits the data for you.
"""

X, y = load_iris(return_X_y=True)

"""
The typical way to split your data into training and test sets is to use the train_test_split function.
You can provide a random seed to make the results reproducible if you need to, but you usually should not,
as the randomness of the split is a good thing. The randomness helps ensure that your model generalizes well.
"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# do something with the train/test data

"""
Another way to split your data is cross validation using the KFold class.
This class allows you to split your data into k folds.
The folds are created randomly, and the class also shuffles the data before splitting it.
"""

kf = KFold(n_splits=5, shuffle=True)

for train_index, test_index in kf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    # do something with the train/test data
