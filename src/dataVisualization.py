from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def displayResults(true_labels, predictions, test_dataset):
    print(f"accuracy: {accuracy_score(true_labels, predictions):.3f}")
    print(classification_report(true_labels, predictions, target_names=list(test_dataset.labels.keys())))

    cm = confusion_matrix(true_labels, predictions)
    ConfusionMatrixDisplay(cm, display_labels=list(test_dataset.labels.keys())).plot()
    plt.show()

