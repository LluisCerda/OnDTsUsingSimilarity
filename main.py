import StDecisionTree as StDecisionTree

if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # Load dataset
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
    
    # Train decision tree
    tree = StDecisionTree(max_depth=3)
    tree.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = tree.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))