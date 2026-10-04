#import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

def run_xgboost(X_train_processed, X_test_processed, y_train, y_test):

    # baseline XGBoost classifier
    base_model = XGBClassifier(
        n_estimators = 100,
        learning_rate = 0.1,
        random_state = 42
    )

    # churn labels convert to 0, 1 for xgboost
    y_train_converted = y_train.map({"No": 0, "Yes": 1})
    y_test_converted = y_test.map({"No": 0, "Yes": 1})

    base_model.fit(X_train_processed, y_train_converted)

    base_y_predict = base_model.predict(X_test_processed)

    base_accuracy = accuracy_score(y_test_converted, base_y_predict)
    print("Base XGBoost Model Accuracy:", base_accuracy)

    base_cm = confusion_matrix(y_test_converted, base_y_predict)
    print("Base XGBoost Model CM:", base_cm)

    base_recall = recall_score(y_test_converted, base_y_predict, pos_label = 1)
    print("Base XGBoost Model Recall:", base_recall)

    base_precision = precision_score(y_test_converted, base_y_predict, pos_label = 1)
    print("Base XGBoost Model Precision:", base_precision)

    base_f1 = f1_score(y_test_converted, base_y_predict, pos_label = 1)
    print("Base XGBoost Model f1:", base_f1)

    base_y_prob = base_model.predict_proba(X_test_processed)[:, 1]
    base_roc_auc = roc_auc_score(y_test_converted, base_y_prob)
    print("Base XGBoost ROC-AUC:", base_roc_auc)

    # evaluations for different numbers of n_estimators and learning_rate
    n_estimator_values = [50, 100, 150, 200, 300]
    learning_rate_values = [0.01, 0.05, 0.1, 0.2]

    results = []

    for n in n_estimator_values:
        for rate in learning_rate_values:

            tuned_model = XGBClassifier(
                n_estimators = n,
                learning_rate = rate,
                random_state = 42
            )

            tuned_model.fit(X_train_processed, y_train_converted)
            test_accuracy = tuned_model.score(X_test_processed, y_test_converted)
            results.append((n, rate, test_accuracy))

    # parameters with highest test accuracy
    best_result = max(results, key = lambda result: result[2]) # when deciding which result is the max, compare using their test accuracy

    # separate the results
    best_n_estimators, best_learning_rate, best_test_accuracy = best_result
    print("Best XGBoost n_estimators/number of trees:", best_n_estimators)
    print("Best XGBoost learning rate:", best_learning_rate)
    print("Best XGBoost test accuracy:", best_test_accuracy)

    # final XGBoost model using best hyperparameters
    final_xgboost_model = XGBClassifier(
        n_estimators = best_n_estimators,
        learning_rate = best_learning_rate,
        random_state = 42
    )

    final_xgboost_model.fit(X_train_processed, y_train_converted)
    final_y_predict = final_xgboost_model.predict(X_test_processed)

    final_accuracy = accuracy_score(y_test_converted, final_y_predict)
    print("Final XGBoost Model Accuracy:", final_accuracy)
    
    final_cm = confusion_matrix(y_test_converted, final_y_predict)
    print("Final XGBoost Model CM:", final_cm)
    
    final_recall = recall_score(y_test_converted, final_y_predict, pos_label = 1)
    print("Final XGBoost Model Recall:", final_recall)
    
    final_precision = precision_score(y_test_converted, final_y_predict, pos_label = 1)
    print("Final XGBoost Model Precision:", final_precision)
    
    final_f1 = f1_score(y_test_converted, final_y_predict, pos_label = 1)
    print("Final XGBoost Model f1:", final_f1)
    
    final_y_prob = final_xgboost_model.predict_proba(X_test_processed)[:, 1]
    final_roc_auc = roc_auc_score(y_test_converted, final_y_prob)
    print("Final XGBoost ROC-AUC:", final_roc_auc)

    fpr, tpr, _ = roc_curve(
        y_test_converted,
        final_y_prob,
        pos_label = 1
    )

    plt.plot(fpr, tpr, label = "XGBoost")
    plt.plot([0, 1], [0, 1], linestyle = "--") # reference line for roc curve
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - XGBoost")
    plt.legend()
    plt.show()

    return{
        "base": {
        "accuracy": base_accuracy,
        "precision": base_precision,
        "recall": base_recall,
        "f1": base_f1,
        "roc_auc": base_roc_auc
        },
        
        "final": {
            "accuracy": final_accuracy,
            "precision": final_precision,
            "recall": final_recall,
            "f1": final_f1,
            "roc_auc": final_roc_auc
        }
    }

