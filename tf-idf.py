''' Classical approach based on TF-IDF and a simple binary classification model. '''

#from util import train_val_test_split
from evaluator import Evaluator

#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import sklearn.linear_model
#from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
import sklearn.model_selection
import sklearn.pipeline
import sklearn.svm
import sklearn.tree


# Load Data
df = pd.read_csv('data/SMSSpamCollection_proprocessed.csv')
df['label'] = df['label'].map({'spam': 1, 'ham': 0})


# Perform Train-Test split
seed = 42
eval_ratio = 0.2
sms_train_val, sms_test, label_train_val, label_test = sklearn.model_selection.train_test_split(
        df['sms'], df['label'], test_size=eval_ratio, random_state=seed)
# No need to split val set due to use of GridSearchCV 
# sms_train, label_train, sms_val, label_val, sms_test, label_test = train_val_test_split(
# df['sms'], df['label'], eval_ratio, seed)


# Create TF-IDF (Term Frequency-Inverse Document Frequency) Matrix
vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(stop_words='english')

# Various classification models
log_regression = sklearn.linear_model.LogisticRegression()
naives_bayes = MultinomialNB()
svm = sklearn.svm.SVC()
decision_tree = sklearn.tree.DecisionTreeClassifier()


#for clf_name, classifier in [('LR', log_regression), ('NB', naives_bayes), ('SVM', svm), ('DT', decision_tree)]:
for clf_name, classifier in [('LR', log_regression)]:

    print('\n', clf_name.upper())
    
    model = sklearn.pipeline.Pipeline([('tfidf', vectorizer),
                                       ('clf', classifier)])
    
    # Hyperparam Tuning
    model2param_grid = {'LR': {'clf__penalty': ['l2', 'elasticnet'],
                               'clf__class_weight': [None, 'balanced']},
                        'NB': {'clf__alpha': [0.1, 0.2, 0.5, 1.]},
                        'SVM': {'clf__C': [0.1, 0.2, 0.5, 1.],
                                'clf__kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']},
                        'DT': {'clf__max_depth': [10, 20, None],
                               'clf__min_samples_split': [0.01, 0.05]}
                        }
    
    
    param_grid = model2param_grid[clf_name]
    grid = GridSearchCV(model, param_grid, cv=5, scoring='f1', verbose=1)

    # Train model
    #model.fit(sms_train, label_train)
    #best_model = model
    
    grid.fit(sms_train_val, label_train_val)
    print('Best Hyperparam config:', grid.best_params_)
    best_model = grid.best_estimator_

    predictions = best_model.predict(sms_test)

    # Evaluation on test data
    evaluator = Evaluator(clf_name)
    evaluator.print_evaluation(predictions, label_test, ['Ham', 'Spam'])

    # TODO: refactor: move to Evaluator
    # Investigate misclassified samples
    print('\nInvestigate Misclassifications:')
    misclassified_idx = np.where(predictions != label_test.to_numpy())
    misclassified_df = pd.concat([label_test, sms_test], axis=1, ignore_index=True).reset_index().loc[misclassified_idx]

    print(misclassified_df.to_string())

# Best Config: SVM with {'clf__C': 1.0, 'clf__kernel': 'linear'}
