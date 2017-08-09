# Parameter Tuning

## SVM Hyperparameters
In the SVM lesson, Katie mentioned optimizing the Gamma and C parameters.

Successfully tuning your algorithm involves searching for a kernel, a gamma value and a C value that minimize prediction error. To tune your SVM vehicle detection model, you can use one of scikit-learn's parameter tuning algorithms.

When tuning SVM, remember that you can only tune the C parameter with a linear kernel. For a non-linear kernel, you can tune C and gamma.

## Parameter Tuning in Scikit-learn
Scikit-learn includes two algorithms for carrying out an automatic parameter search:

- GridSearchCV
- RandomizedSearchCV

GridSearchCV exhaustively works through multiple parameter combinations, cross-validating as it goes. The beauty is that it can work through many combinations in only a couple extra lines of code.

For example, if I input the values C:[0.1, 1, 10] and gamma:[0.1, 1, 10], gridSearchCV will train and cross-validate every possible combination of (C, gamma): (0.1, 0.1), (0.1, 1), (0.1, 10), (1, .1), (1, 1), etc.

RandomizedSearchCV works similarly to GridSearchCV except RandomizedSearchCV takes a random sample of parameter combinations. RandomizedSearchCV is faster than GridSearchCV since RandomizedSearchCV uses a subset of the parameter combinations.

## Cross-validation with GridSearchCV
GridSearchCV uses 3-fold cross validation to determine the best performing parameter set. GridSearchCV will take in a training set and divide the training set into three equal partitions. The algorithm will train on two partitions and then validate using the third partition. Then GridSearchCV chooses a different partition for validation and trains with the other two partitions. Finally, GridSearchCV uses the last remaining partition for cross-validation and trains with the other two partitions.

By default, GridSearchCV uses accuracy as an error metric by averaging the accuracy for each partition. So for every possible parameter combination, GridSearchCV calculates an accuracy score. Then GridSearchCV will choose the parameter combination that performed the best.

## Scikit-learn Cross Validation Example
Here's an example from the sklearn documentation for implementing GridSearchCV:

    parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
    svr = svm.SVC()
    clf = grid_search.GridSearchCV(svr, parameters)
    clf.fit(iris.data, iris.target)

Let's break this down line by line.

    parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]} 

A dictionary of the parameters, and the possible values they may take. In this case, they're playing around with the kernel (possible choices are 'linear' and 'rbf'), and C (possible choices are 1 and 10).

Then a 'grid' of all the following combinations of values for (kernel, C) are automatically generated:

('rbf', 1)	('rbf', 10)
('linear', 1)	('linear', 10)

Each is used to train an SVM, and the performance is then assessed using cross-validation.

    svr = svm.SVC() 

This looks kind of like creating a classifier, just like we've been doing since the first lesson. But note that the "clf" isn't made until the next line--this is just saying what kind of algorithm to use. Another way to think about this is that the "classifier" isn't just the algorithm in this case, it's algorithm plus parameter values. Note that there's no monkeying around with the kernel or C; all that is handled in the next line.

    clf = grid_search.GridSearchCV(svr, parameters) 

This is where the first bit of magic happens; the classifier is being created. We pass the algorithm (svr) and the dictionary of parameters to try (parameters) and it generates a grid of parameter combinations to try.

    clf.fit(iris.data, iris.target) 

And the second bit of magic. The fit function now tries all the parameter combinations, and returns a fitted classifier that's automatically tuned to the optimal parameter combination. You can now access the parameter values via clf.best_params_.
