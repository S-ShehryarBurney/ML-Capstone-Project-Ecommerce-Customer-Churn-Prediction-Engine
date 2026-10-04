#import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

def run_logistic_regression(X_train_processed, X_test_processed, y_train, y_test):
    model = LogisticRegression(max_iter = 1000)
    model.fit(X_train_processed, y_train)

    y_pred = model.predict(X_test_processed)
    print(y_pred[:20])

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    cm = confusion_matrix(y_test, y_pred)
    print("cm:", cm)

    recall = recall_score(y_test, y_pred, pos_label = "Yes")
    print("Recall Score:", recall)

    precision = precision_score(y_test, y_pred, pos_label = "Yes")
    print("Precision Score:", precision)

    f1 = f1_score(y_test, y_pred, pos_label = "Yes")
    print("F1-Score:", f1)

    # model's predicted probability of churn = yes
    y_prob = model.predict_proba(X_test_processed)[:, 1]
    print("Model's Predicted Probability for Churn = Yes:", y_prob)

    roc_auc = roc_auc_score(y_test, y_prob)
    print("ROC_AUC Score:", roc_auc)

    fpr, tpr, _ = roc_curve(y_test, y_prob, pos_label = "Yes")
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - LogisticRegression")
    plt.show()

    # L1 Regularization
    L1_model = LogisticRegression(
        l1_ratio = 1,
        solver = "liblinear",
        max_iter = 1000
    )

    L1_model.fit(X_train_processed, y_train)

    L1_y_pred = L1_model.predict(X_test_processed)

    L1_accuracy = accuracy_score(y_test, L1_y_pred)
    print("L1_accuracy:", L1_accuracy)

    L1_cm = confusion_matrix(y_test, L1_y_pred)
    print("L1_cm:", L1_cm)

    L1_recall = recall_score(y_test, L1_y_pred, pos_label = "Yes")
    print("L1 Recall Score:", L1_recall)

    L1_precision = precision_score(y_test, L1_y_pred, pos_label = "Yes")
    print("L1 Precision Score:", L1_precision)

    L1_f1 = f1_score(y_test, L1_y_pred, pos_label = "Yes")
    print("L1 F1-Score:", L1_f1)

    L1_y_prob = L1_model.predict_proba(X_test_processed)[:, 1]
    print("L1 Model's Predicted Probability for Churn = Yes:", L1_y_prob)

    L1_roc_auc = roc_auc_score(y_test, L1_y_prob)
    print("L1 ROC_AUC Score:", L1_roc_auc)

    # L2 Regularization
    L2_model = LogisticRegression(
        l1_ratio = 0,
        max_iter = 1000
    )

    L2_model.fit(X_train_processed, y_train)

    L2_y_pred = L2_model.predict(X_test_processed)

    L2_accuracy = accuracy_score(y_test, L2_y_pred)
    print("L2_accuracy:", L2_accuracy)

    L2_cm = confusion_matrix(y_test, L2_y_pred)
    print("L2_cm:", L2_cm)

    L2_recall = recall_score(y_test, L2_y_pred, pos_label = "Yes")
    print("L2 Recall Score:", L2_recall)

    L2_precision = precision_score(y_test, L2_y_pred, pos_label = "Yes")
    print("L2 Precision Score:", L2_precision)

    L2_f1 = f1_score(y_test, L2_y_pred, pos_label = "Yes")
    print("L2 F1-Score:", L2_f1)

    L2_y_prob = L2_model.predict_proba(X_test_processed)[:, 1]
    print("L2 Model's Predicted Probability for Churn = Yes:", L2_y_prob)

    L2_roc_auc = roc_auc_score(y_test, L2_y_prob)
    print("L2 ROC_AUC Score:", L2_roc_auc)

    return{
        "L1": {
            "accuracy": L1_accuracy,
            "precision": L1_precision,
            "recall": L1_recall,
            "f1": L1_f1,
            "roc_auc": L1_roc_auc
        },

        "L2": {
            "accuracy": L2_accuracy,
            "precision": L2_precision,
            "recall": L2_recall,
            "f1": L2_f1,
            "roc_auc": L2_roc_auc
        }
    }
    