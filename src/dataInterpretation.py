from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from pathlib import Path
from globalVariables import MODELS_PATH
import matplotlib.pyplot as plt
import torch
import os

def getConsistentAccuracyFormat(acc, *, with_space: bool = False) -> str:
    return f"accuracy: {acc:.3f}" if not with_space else f"accuracy: {acc:.3f}\n\n"

def overwriteModelDataInFolder(model, acc, report, test_name, txt_path, *, run_dir):
    # overwrite txt_path file
    with open(txt_path, "w") as f:
        f.write(getConsistentAccuracyFormat(acc, with_space=True))
        f.write(report)

    # save kernel_size and is_deep
    txt_arch = os.path.join(run_dir, f"{test_name}_architecture.txt" ) # results text that will contain general report
    with open(txt_arch, "w") as f:
        f.write(f"kernel_size: {model.kernel_size} is_deep: {model.is_deep}")

    # overwrite confusion matrix
    plt.savefig(os.path.join(run_dir, f"{test_name}_confusion_matrix.png"))

    # overwrite model
    torch.save(model.state_dict(), os.path.join(run_dir, f"{test_name}_model.pt"))

def interpretResults(agent, test_dataset, test_loader, *, test_name="model", save_only_if_better: bool = False, display_confusion_matrix: bool = False):

    true_labels, predictions = agent.evaluate(test_loader) # evaluate model against test_loader

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

    if save_only_if_better: # will save if there's no other entry, overwrite if the new entry is better than prev
        if Path(txt_path).is_file():
            with open(txt_path) as f:
                old_acc = float(f.readline().split(" ")[1]) # formatting should stay consistent with a space seperating accuracy and the number
                
            if new_acc > old_acc: # rewrite old stuff
                print(f"overwriting old {os.path.join(run_dir, test_name)} since accuracy is better this run")
                overwriteModelDataInFolder(agent.model, new_acc, report, test_name, txt_path, run_dir=run_dir)
            else:
                print(f"skipping {test_name}, new acc {new_acc:.3f} did not beat old acc {old_acc:.3f}")
        else:
            print(f"adding new model and figures entry to {os.path.join(run_dir, test_name)}")
            overwriteModelDataInFolder(agent.model, new_acc, report, test_name, txt_path, run_dir=run_dir)

    plt.close()
