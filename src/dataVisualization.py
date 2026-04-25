from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os

def displayResults(true_labels, predictions, test_dataset, test_name):
    figures_dir = "figures"
    run_dir = os.path.join(figures_dir, test_name)
    os.makedirs(run_dir, exist_ok=True)

    accuracy = accuracy_score(true_labels, predictions)
    report = classification_report(true_labels, predictions, target_names=list(test_dataset.labels.keys()))

    print(f"accuracy: {accuracy:.3f}")
    print(report)

    txt_path = os.path.join(run_dir, f"{test_name}_results.txt")
    with open(txt_path, "w") as f:
        f.write(f"accuracy: {accuracy:.3f}\n\n")
        f.write(report)

    cm = confusion_matrix(true_labels, predictions)
    disp = ConfusionMatrixDisplay(cm, display_labels=list(test_dataset.labels.keys()))
    disp.plot(xticks_rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, f"{test_name}_confusion_matrix.png"), dpi=150)
    plt.close()
