from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from pathlib import Path
from filePaths import MODELS_PATH
import matplotlib.pyplot as plt
import os

def getConsistentAccuracyFormat(acc, *, with_space: bool = False) -> str:
    return f"accuracy: {acc:.3f}" if not with_space else f"accuracy: {acc:.3f}\n\n"

def overwriteModelDataInFolder(acc, report, test_name, txt_path, *, run_dir):
    with open(txt_path, "w") as f:
        f.write(getConsistentAccuracyFormat(acc, with_space=True))
        f.write(report)
        plt.savefig(os.path.join(run_dir, f"{test_name}_confusion_matrix.png"))

def interpretResults(true_labels, predictions, test_dataset, *, test_name="model", save_only_if_better: bool = False, display_confusion_matrix: bool = False):
    new_acc = accuracy_score(true_labels, predictions)
    report = classification_report(true_labels, predictions, target_names=list(test_dataset.labels.keys()))

    print(getConsistentAccuracyFormat(new_acc))
    print(report)

    # create confusion matrix w/data
    cm = confusion_matrix(true_labels, predictions)
    disp = ConfusionMatrixDisplay(cm, display_labels=list(test_dataset.labels.keys()))
    disp.plot(xticks_rotation=45)
    plt.tight_layout()

    if display_confusion_matrix: # display confusion matrix at the end of model training
        plt.show()

    path = MODELS_PATH
    run_dir = os.path.join(path, test_name)
    os.makedirs(run_dir, exist_ok=True)

    txt_path = os.path.join(run_dir, f"{test_name}_results.txt") # results text that will contain general report

    if save_only_if_better and Path(txt_path).is_file(): # save if accuracy is better overall, will only overwrite if test_name exists and path exists
        with open(txt_path) as f:
            old_acc = float(f.readline().split(" ")[1]) # formatting should stay consistent with a space seperating accuracy and the number
            
        if new_acc >= old_acc: # rewrite old stuff
            print(f"overwriting old {run_dir}/{txt_path} since accuracy is better this run")
            overwriteModelDataInFolder(new_acc, report, test_name, txt_path, run_dir=run_dir)
    else:
        print(f"overwriting old {run_dir}/{txt_path}")
        overwriteModelDataInFolder(new_acc, report, test_name, txt_path, run_dir=run_dir)
    plt.close()
