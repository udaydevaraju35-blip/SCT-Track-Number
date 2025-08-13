import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def decision_tree_bank():
    print("=== Decision Tree Model: Bank Subscription Prediction ===")

    dataset_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional-full.csv"
    try:
        data = pd.read_csv(dataset_url, sep=";")
        print(f"[OK] Dataset loaded with {len(data)} rows and {len(data.columns)} columns.")
    except Exception as e:
        print(f"[ERROR] Could not load dataset: {e}")
        return

    if "y" not in data.columns:
        print("[ERROR] Target column 'y' not found in dataset.")
        return

    data["y"] = data["y"].map({"yes": 1, "no": 0})

    if "duration" in data.columns:
        data.drop(columns="duration", inplace=True)

    X = pd.get_dummies(data.drop("y", axis=1), drop_first=True)
    y = data["y"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=7, stratify=y
    )

    model = DecisionTreeClassifier(max_depth=5, random_state=7)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    plt.figure(figsize=(26, 12))
    plot_tree(
        model,
        feature_names=X.columns,
        class_names=["No", "Yes"],
        filled=True,
        rounded=True,
        fontsize=9,
        max_depth=3
    )
    plt.title("Decision Tree - Bank Subscription Prediction (Top 3 Levels)")
    plt.savefig("decision_tree_plot.png")
    plt.close()
    print("[OK] Decision tree saved as 'decision_tree_plot.png'.")

if __name__ == "__main__":
    decision_tree_bank()
