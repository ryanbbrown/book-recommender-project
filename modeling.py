# data split
from sklearn.preprocessing import LabelBinarizer

lb = LabelBinarizer()
X, y = df.text, df.label
y = lb.fit_transform(y).reshape(len(y),)
titles = df.title.tolist()


# modeling loop
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

models = [LogisticRegression(solver='liblinear', max_iter=300),
          SVC(C=1.0, kernel='linear', degree=3, gamma='auto'), MultinomialNB(),
          KNeighborsClassifier(), RidgeClassifier(),
          SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3)]

accuracy_mean, accuracy_std, precision_mean, precision_std = [], [], [], []

for model in models:
    pipe = Pipeline([('cleaner', clean_transformer()),
                 ('vectorizer', bow_vector),
                 ('classifier', model)])
    
    accuracy = cross_val_score(estimator=pipe, X=X, y=y, groups=titles, cv=GroupKFold(), scoring='accuracy')
    precision = cross_val_score(estimator=pipe, X=X, y=y, groups=titles, cv=GroupKFold(), scoring='precision')
    
    accuracy_mean.append(np.mean(accuracy))
    accuracy_std.append(np.std(accuracy))
    precision_mean.append(np.mean(precision))
    precision_std.append(np.std(precision))
    
    
# hyperparameter tuning
classifier = KNeighborsClassifier()

pipe = Pipeline([('cleaner', clean_transformer()),
                 ('vectorizer', tfidf_vector),
                 ('classifier', classifier)])

params = {'classifier__n_neighbors':[1,3,5,7,9,11],
         'classifier__weights':['uniform','distance'],
         'classifier__leaf_size':[10,30,50,100,200],
         'classifier__p':[1,2]}

clf = GridSearchCV(estimator=pipe, param_grid=params, cv=GroupKFold(), verbose=2, scoring='precision')
clf.fit(X, y, groups=titles)

print(clf.best_params_)
print(clf.best_score_)
