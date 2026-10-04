import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

def run_random_forest(X_train_processed, X_test_processed, y_train, y_test):

    # Baseline Random Forest Classifier
    model = RandomForestClassifier(
        n_estimators = 100,
        random_state = 42
    )

    model.fit(X_train_processed, y_train)

    model_y_pred = model.predict(X_test_processed)

    baseline_accuracy = accuracy_score(y_test, model_y_pred)
    #print("\nBaseline Random Forest Accuracy:", baseline_accuracy)

    baseline_confusion_matrix = confusion_matrix(y_test, model_y_pred)
    print("\nBaseline Random Forest Confusion Matrix:", baseline_confusion_matrix)

    baseline_recall = recall_score(y_test, model_y_pred, pos_label = "Yes")
    print("\nBaseline Random Forest Recall:", baseline_recall)

    baseline_precision = precision_score(y_test, model_y_pred, pos_label = "Yes")
    print("\nBaseline Random Forest Precision:", baseline_precision)

    baseline_f1 = f1_score(y_test, model_y_pred, pos_label = "Yes")
    print("\nBaseline Random Forest F1-Score:", baseline_f1)

    baseline_y_prob = model.predict_proba(X_test_processed)[:, 1]
    print("\nBaseline Random Forest ROC-AUC:", roc_auc_score(y_test, baseline_y_prob))

    # Comparing training and test accuracy
    train_accuracy = model.score(X_train_processed, y_train)
    test_accuracy = model.score(X_test_processed, y_test)

    print("\nRandom Forest Train Accuracy:", train_accuracy)
    print("Random Forest Test Accuracy:", test_accuracy)

    # Test accuracy comparison for different n_estimators 
    estimators = [20, 50, 80, 100, 150, 300]
    test_accuracies = []

    for n in estimators:
        estimator_model = RandomForestClassifier(
            n_estimators = n,
            random_state = 42
        )

        estimator_model.fit(X_train_processed, y_train)

        test_accuracies.append(estimator_model.score(X_test_processed, y_test))

    # test accuracy plotting against number of trees
    plt.plot(estimators, test_accuracies, marker = "o")
    plt.xlabel("Number of Trees")
    plt.ylabel("Test Accuracy")
    plt.title("RF Test Accuracy Against Number of Trees")
    plt.show()

    # number of trees with highest test accuracy
    best_estimator = estimators[test_accuracies.index(max(test_accuracies))]
    best_test_accuracy = max(test_accuracies)

    print("\nBest Number of Trees:", best_estimator)
    print("Best Test Accuracy:", best_test_accuracy)

    # Random forest classifier with the best number of trees (n_estimators)
    final_model = RandomForestClassifier(
        n_estimators = best_estimator,
        random_state = 42
    )

    final_model.fit(X_train_processed, y_train)

    final_y_pred = final_model.predict(X_test_processed)

    # Evaluations
    final_accuracy = accuracy_score(y_test, final_y_pred)
    print("\nFinal Random Forest Accuracy:", final_accuracy)

    final_confusion_matrix = confusion_matrix(y_test, final_y_pred)
    print("\nFinal Random Forest Confusion Matrix:", final_confusion_matrix)

    final_recall = recall_score(y_test, final_y_pred, pos_label = "Yes")
    print("\nFinal Random Forest Recall:", final_recall)

    final_precision = precision_score(y_test, final_y_pred, pos_label = "Yes")
    print("\nFinal Random Forest Precision:", final_precision)

    final_f1 = f1_score(y_test, final_y_pred, pos_label = "Yes")
    print("\nFinal Random Forest F1-Score:", final_f1)

    final_y_prob = final_model.predict_proba(X_test_processed)[:, 1]
    final_roc_auc_score = roc_auc_score(y_test, final_y_prob)
    print("\nFinal Random Forest ROC-AUC:", final_roc_auc_score)

    fpr, tpr, _ = roc_curve(y_test, final_y_prob, pos_label = "Yes")
    plt.plot(fpr, tpr, label = "Random Forest")
    plt.plot([0, 1], [0, 1], linestyle = "--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Random Forest")
    plt.legend()
    plt.show()

    comparison = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
            "Baseline": [baseline_accuracy, baseline_precision, baseline_recall, baseline_f1, roc_auc_score(y_test, baseline_y_prob)],
            "Final": [final_accuracy, final_precision, final_recall, final_f1, final_roc_auc_score]
        })
    
    comparison["Baseline"] = comparison["Baseline"] * 100
    comparison["Final"] = comparison["Final"] * 100
    
    print("\nBaseline vs Final Random Forest:")
    print(comparison.to_string(index = False, float_format = "%.2f%%"))
