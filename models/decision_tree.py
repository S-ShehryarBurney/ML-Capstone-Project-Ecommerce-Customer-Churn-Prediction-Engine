import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

def run_decision_tree(X_train_processed, X_test_processed, y_train, y_test):

    # Unconstrained Decision Tree Classifier
    # Allow the tree to grow freely to analyze overfitting
    model = DecisionTreeClassifier(random_state = 42)
    model.fit(X_train_processed, y_train)

    # Unconstrained Model Evaluation
    model_y_pred = model.predict(X_test_processed)

    unconstrained_accuracy = accuracy_score(y_test, model_y_pred)
    print("\nUnconstrained Decision Tree Accuracy:", unconstrained_accuracy)

    unconstrained_confusion_matrix = confusion_matrix(y_test, model_y_pred)
    print("\nUnconstrained Decision Tree Confusion Matrix:", unconstrained_confusion_matrix)

    unconstrained_recall = recall_score(y_test, model_y_pred, pos_label = "Yes")
    print("\nUnconstrained Decision Tree Recall:", unconstrained_recall)

    unconstrained_precision = precision_score(y_test, model_y_pred, pos_label = "Yes")
    print("\nUnconstrained Decision Tree Precision:", unconstrained_precision)

    unconstrained_f1 = f1_score(y_test, model_y_pred, pos_label = "Yes")
    print("\nUnconstrained Decision Tree F1-Score:", unconstrained_f1)

    model_y_prob = model.predict_proba(X_test_processed)[:, 1]
    unconstrained_roc_auc = roc_auc_score(y_test, model_y_prob)
    print("Unconstrained Decision Tree ROC-AUC:", unconstrained_roc_auc)

    # Compare training and test accuracy to assess overfitting
    train_accuracy = model.score(X_train_processed, y_train)
    test_accuracy = model.score(X_test_processed, y_test)

    print("Decision Tree Classifier Train Accuracy:", train_accuracy)
    print("Decision Tree Classifier Test Accuracy:", test_accuracy)

    # Cost Complexity Pruning Path
    pruning_path = model.cost_complexity_pruning_path(X_train_processed, y_train)

    ccp_alphas = pruning_path.ccp_alphas
    #impurities = pruning_path.impurities

    print("\nNumber of CCP-Alphas:", len(ccp_alphas))
    #print("CCP-Alphas:", ccp_alphas)

    # Training and test accuracy at different tree depths
    depths = range(1, 21)

    train_accuracies = []
    test_accuracies = []

    for depth in depths:
        depth_model = DecisionTreeClassifier(
            max_depth = depth,
            random_state = 42
        )

        depth_model.fit(X_train_processed, y_train)

        train_accuracies.append(depth_model.score(X_train_processed, y_train))
        test_accuracies.append(depth_model.score(X_test_processed, y_test))

    plt.plot(depths, train_accuracies, label = "Train Accuracy")
    plt.plot(depths, test_accuracies, label = "Test Accuracy")

    plt.xlabel("Tree Depth")
    plt.ylabel("Accuracy")
    plt.title("Decision Tree Accuracy vs Tree Depth")
    plt.legend()
    plt.show()

    # Tree depth with the highest test accuracy
    best_depth = depths[test_accuracies.index(max(test_accuracies))]
    best_test_accuracy = max(test_accuracies)

    print("\nBest Tree Depth:", best_depth)
    print("Best Test Accuracy:", best_test_accuracy)

    # Evaluate training and test accuracy for different ccp-alpha values
    pruned_train_accuracies = []
    pruned_test_accuracies = []

    for ccp_alpha in ccp_alphas[:-1]: # Excluding last alpha value which completely prunes the tree
        pruned_model = DecisionTreeClassifier(
            random_state = 42,
            ccp_alpha = ccp_alpha
        )

        pruned_model.fit(X_train_processed, y_train)

        pruned_train_accuracies.append(pruned_model.score(X_train_processed, y_train))
        pruned_test_accuracies.append(pruned_model.score(X_test_processed, y_test))

    # identifying ccp-alpha tha produces the highest test accuracy
    best_alpha_index = pruned_test_accuracies.index(max(pruned_test_accuracies))

    best_alpha = ccp_alphas[best_alpha_index]
    best_pruned_test_accuracy = pruned_test_accuracies[best_alpha_index]

    print("\nBest CCP-Alpha:", best_alpha)
    print("Best Pruned Test Accuracy:", best_pruned_test_accuracy)

    # Final Decision Tree using best ccp-alpha value
    pruned_model = DecisionTreeClassifier(
        random_state = 42,
        ccp_alpha = best_alpha
    )

    pruned_model.fit(X_train_processed, y_train)

    # Pruned Model Evaluation
    pruned_y_pred = pruned_model.predict(X_test_processed)

    pruned_accuracy = accuracy_score(y_test, pruned_y_pred)
    print("\nPruned Decision Tree Accuracy:", pruned_accuracy)

    pruned_confusion_matrix = confusion_matrix(y_test, pruned_y_pred)
    print("\nPruned Decision Tree Confusion Matrix:", pruned_confusion_matrix)

    pruned_recall = recall_score(y_test, pruned_y_pred, pos_label = "Yes")
    print("\nPruned Decision Tree Recall:", pruned_recall)

    pruned_precision = precision_score(y_test, pruned_y_pred, pos_label = "Yes")
    print("\nPruned Decision Tree Precision:", pruned_precision)

    pruned_f1 = f1_score(y_test, pruned_y_pred, pos_label = "Yes")
    print("\nPruned Decision Tree F1-Score:", pruned_f1)

    pruned_y_prob = pruned_model.predict_proba(X_test_processed)[:, 1]
    pruned_roc_auc = roc_auc_score(y_test, pruned_y_prob)
    print("\nPruned Decision Tree ROC-AUC:", pruned_roc_auc)

    fpr, tpr, _ = roc_curve(y_test, pruned_y_prob, pos_label = "Yes")
    plt.plot(fpr, tpr, label = "Pruned Decision Tree")

    # Draw a reference line. A useful ROC curve should be above this line
    plt.plot([0, 1], [0, 1], linestyle = "--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Pruned Decision Tree")
    plt.legend()
    plt.show()

    comparison = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
            "unconstrained": [unconstrained_accuracy, unconstrained_precision, unconstrained_recall, unconstrained_f1, unconstrained_roc_auc],
            "pruned": [pruned_accuracy, pruned_precision, pruned_recall, pruned_f1, pruned_roc_auc]
        })
    
    comparison["unconstrained"] = comparison["unconstrained"] * 100
    comparison["pruned"] = comparison["pruned"] * 100
    
    print("\nUnconstrained vs Pruned Decision Tree:")
    print(comparison.to_string(index = False, float_format = "%.2f%%"))
