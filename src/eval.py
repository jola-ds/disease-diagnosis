import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.calibration import calibration_curve
import numpy as np

def bootstrap_accuracy_ci(model, X_test, y_test, n_bootstrap=1000, ci=95):
    """Bootstrap accuracy with confidence intervals."""
    rng = np.random.default_rng(42)
    accs = []
    n = len(y_test)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        X_resample = X_test.iloc[idx] if hasattr(X_test, "iloc") else X_test[idx]
        y_resample = y_test.iloc[idx] if hasattr(y_test, "iloc") else y_test[idx]
        y_pred = model.predict(X_resample)
        accs.append(accuracy_score(y_resample, y_pred))
    lower = np.percentile(accs, (100-ci)/2)
    upper = np.percentile(accs, 100 - (100-ci)/2)
    print(f"[Bootstrap Accuracy] Mean={np.mean(accs):.3f}, {ci}% CI=({lower:.3f}, {upper:.3f})")
    return np.mean(accs), lower, upper

def evaluate_model(model, X_test, y_test, class_names):
    """Print metrics, plot confusion matrix, and calibration curve."""
    y_pred = model.predict(X_test)

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("plots/confusion_matrix_final.png")
    plt.show()

    # Calibration Curve (example: class 0 only)
    prob_pos = model.predict_proba(X_test)[:, 0]
    prob_true, prob_pred = calibration_curve(y_test == 0, prob_pos, n_bins=10)
    plt.plot(prob_pred, prob_true, marker='o')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.title("Calibration Curve (Class 0)")
    plt.xlabel("Predicted Probability")
    plt.ylabel("True Probability")
    plt.tight_layout()
    plt.savefig("plots/calibration_curve_final.png")
    plt.show()

    # Bootstrap CI
    bootstrap_accuracy_ci(model, X_test, y_test)
print("done")