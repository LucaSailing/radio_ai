import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from radio_ai_package.ml_logic.registry import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report

########################## Getting threhold from user ##########################
def get_user_threshold(default=0.5):
    """
    Prompts the user to enter a classification threshold between 0.0 and 1.0.
    Falls back to `default` if the user presses Enter or inputs invalid data.
    """
    while True:
        user_input = input(f"Enter probability threshold (0.0 to 1.0) [default: {default}]: ").strip()

        # Fallback to default if input is empty
        if not user_input:
            print(f"-> Using default threshold: {default}")
            return default

        try:
            threshold = float(user_input)
            if 0.0 <= threshold <= 1.0:
                print(f"-> Selected threshold: {threshold}")
                return threshold
            else:
                print("Error: Threshold must be between 0.0 and 1.0. Please try again.")
        except ValueError:
            print("Error: Invalid number format. Please enter a float (e.g., 0.35).")


######################### Defyining y_test, preds, x_test ######################
def get_predictions (test_ds):
    model = load_model()
    preds = model.predict(test_ds)
    return preds

def get_x_test(test_ds):
    '''Extract all image batches (x) into stacked arrays'''
    X_test = np.concatenate([x.numpy() for x, _ in test_ds)], axis=0)
    print("X_test shape:", X_test.shape)
    return X_test

def get_y_test(test_ds):
    '''Extract all image label batches (y) into stacked arrays'''
    y_test = np.concatenate([y for x, y in test_ds)], axis=0)
    print("X\y_test shape:", y_test.shape)
    return y_test

def get_binary_predictions(preds):
    '''flattening preds into 1 dimension and turns them into a binary class
    taking into account a threshold value for the probabilities of an X-ray to show a fracture'''
    threshold = get_user_threshold(default=0.5)
    binary_preds = (preds.flatten() > threshold).astype(int)
    return binary_preds

############################### Confusion matrix ###############################
def get_confusion_matrix(y_test, preds):
    # Confusion matrix
    y_test = y_test # list of actual truths
    preds = get_binary_predictions (preds) # list of predictions

    results_df = pd.DataFrame({"actual": y_test,
                           "predicted": preds}) #Store results in a dataframe

    confusion_matrix = pd.crosstab(index= results_df['actual'],
                               columns = results_df['predicted'])
    return confusion_matrix

def confusion_matrix_display_predicted(y_test, preds):
    preds = get_binary_predictions (preds)
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        preds,
        display_labels=['Normal (0)', 'Fracture (1)'],
        cmap=plt.cm.Blues
    )

    plt.title("Confusion Matrix - Test Set")
    plt.show()
    return None


################################ Comparing metrics ##############################
def comparing_metrics_predictions(y_test, preds):
    y_true = y_test
    y_pred_binary = get_binary_predictions(preds)

    accuracy = round(accuracy_score(y_true, y_pred_binary), 2) # Accuracy
    precision = round(precision_score(y_true, y_pred_binary), 2) # Precision
    recall = round(recall_score(y_true, y_pred_binary), 2) # Recall
    f1 = round(f1_score(y_true, y_pred_binary), 2) # F1 score

    print(f'Accuracy = {accuracy}') # Accuracy
    print(f'Precision = {precision}')
    print(f'Recall = {recall}')
    print(f'F1 score = {f1}')
    return accuracy, precision, recall, f1

############################# Classification report ############################
'''Classification report performed using MY predictions:
It evaluates only the data on the test set;
The prediction source is only a single model trained once on X_train;
Higher risk of luck based on how hte split landed'''

def get_classification_report (y_test, preds):
    y_pred_binary = get_binary_predictions(preds)

    # Print the classification report
    classif = (classification_report(
        y_test,
        y_pred_binary,
        target_names=['Normal (0)', 'fracture (1)'],
        digits = 2
    ))
    print(classif)
    return classif


############################# Precision Recall Curve ###########################
def pr_curve (y_test, preds):
    y_pred_binary = get_binary_predictions(preds)
    return precision_recall_curve(y_test, y_pred_binary)

def plot_pr_curve(y_test, preds):
    '''Pass raw continuous probabilities, NOT binary 0/1 predictions)'''
    # Ensure inputs are 1D arrays
    y_true_1d = y_test.flatten()
    y_probs_1d = preds.flatten()

    # Create display and plot
    fig, ax = plt.subplots(figsize=(7, 5))
    disp = PrecisionRecallDisplay.from_predictions(
        y_true_1d,
        y_probs_1d,
        name="CNN Model",
        ax=ax
    )
    plt.title("Precision-Recall Curve")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()

###################################### ROC-AUC #################################
def get_roc_auc_analysis(y_test, preds):
    """
    Computes ROC-AUC, calculates the optimal decision threshold via Youden's J statistic,
    and plots the ROC curve.

    y_true: Ground truth binary labels (0 or 1)
    y_probs: Continuous probability predictions (0.0 to 1.0)
    """
    # 1. Flatten inputs to ensure 1D compatibility
    y_true_1d = np.array(y_test).flatten()
    y_probs_1d = np.array(preds).flatten()

    # 2. Extract metrics and thresholds
    fpr, tpr, thresholds = roc_curve(y_true_1d, y_probs_1d)
    auc_score = roc_auc_score(y_true_1d, y_probs_1d)

    # 3. Compute Youden's J statistic (J = Sensitivity + Specificity - 1 = TPR - FPR)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold_j = thresholds[best_idx]

    # Print summary metrics
    print(f"AUC Score: {auc_score:.4f}")
    print(f"Optimal Threshold (Youden's J): {best_threshold_j:.4f}")
    print(f"At this threshold -> TPR (Sensitivity): {tpr[best_idx]:.4f}, FPR (1-Specificity): {fpr[best_idx]:.4f}")

    # 4. Plot the ROC curve
    plt.figure(figsize=(8, 6))

    # Main ROC curve line
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {auc_score:.3f})')

    # Diagonal baseline (random classifier)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier (AUC = 0.50)')

    # Highlight optimal point
    plt.scatter(
        fpr[best_idx],
        tpr[best_idx],
        color='red',
        s=100,
        zorder=5,
        label=f'Optimal Threshold = {best_threshold_j:.3f}'
    )

    # Formatting plot labels and boundaries
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Recall / Sensitivity)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)

    plt.show()

    return fpr, tpr, thresholds, auc_score, best_threshold_j
