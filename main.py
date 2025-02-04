import StDecisionTree as StDecisionTree
import graphviz as graphviz
import pandas as pd

def load_dataset():
    data = load_iris()
    return data

def visualize_dataframe(data):
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    print(df)

if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    #LOAD
    data = load_dataset()

    #visualize_dataframe(data)

    #SPLIT
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
    
    # Train decision tree
    tree = StDecisionTree.StDecisionTree()
    tree.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = tree.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))

    dot = tree.visualize_tree_graphviz()
    dot.render("decision_tree", format="png", view=True)
    