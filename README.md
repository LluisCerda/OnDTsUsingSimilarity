# Similarity Decision Tree (Gower Distance, N-ary, NaN Handling)

This project implements a custom decision tree algorithm, `SimilarityDecisionTree_D18`, which can function as both a classifier and a regressor. Unlike traditional decision trees that use impurity measures like Gini or entropy, this tree uses Gower similarity to a randomly selected prototype to determine splits. It uniquely supports n-ary splits (more than two children per node) and robustly handles missing values (NaNs) directly within its similarity calculations.

## Key Features

*   **Gower Similarity:** Utilizes Gower's general coefficient of similarity to handle mixed data types (numerical and categorical) naturally.
*   **Prototype-Based Splitting:** At each node, a random sample (prototype) is selected. Other samples in the node are split based on their Gower similarity to this prototype.
*   **N-ary Splits via Iterative Mean Splitting:**
    *   Supports `n_children_per_node` (where `n >= 2`).
    *   Thresholds are determined by iteratively splitting the largest available group of samples by its mean similarity value until the target number of child groups is approached or splitting conditions are no longer met.
    *   If `n_children_per_node=2`, this method effectively splits the node by the overall mean of similarities, mimicking a common binary split strategy.
*   **Missing Value (NaN) Handling:**
    *   NaNs are not imputed.
    *   During Gower similarity calculation between a sample and a prototype, if a feature value is NaN in either, that feature is ignored for that specific comparison.
    *   The similarity score is then a weighted sum over only the mutually present features.
*   **Numerical Data Normalization:** Numeric features are automatically min-max scaled (0-1 range) during training. This scaling is applied to new data during prediction. NaNs are handled appropriately during normalization.
*   **Feature Weighting:** Allows users to provide weights for different features, influencing their importance in the similarity calculation.
*   **Parallelization:** Leverages `joblib` for parallel tree building, potentially speeding up training on multi-core processors for large datasets.
*   **Classifier & Regressor:** Can be used for both classification and regression tasks by setting the `isClassifier` flag.

## Requirements

*   Python 3.x
*   NumPy
*   joblib

You can install them using pip:
```bash
pip install numpy joblib
```

## How to Use

1.  Place `SimilarityDecisionTree_D18.py` in your project directory or ensure it's in your Python path.
2.  Import the class: `from SimilarityDecisionTree_D18 import SimilarityDecisionTree_D18`

### Example

```python
import numpy as np
from SimilarityDecisionTree_D18 import SimilarityDecisionTree_D18
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

# --- Sample Data (Illustrative) ---
# Features:
# 0: Numerical (with NaN)
# 1: Categorical (0, 1, 2)
# 2: Numerical
# 3: Categorical (0, 1) (with NaN)
X = np.array([
    [10, 0, 100, 0],
    [12, 1, 110, 1],
    [np.nan, 0, 105, 0], # Sample with NaN in numerical
    [15, 2, 120, np.nan], # Sample with NaN in categorical
    [11, 1, 90, 1],
    [13, 0, 115, 0],
    [16, 2, 125, 1],
    [10, 1, 95, np.nan],
    [14, 0, 100, 0],
    [12, 2, 130, 1]
], dtype=object) # Use dtype=object if mixing types and NaNs, then convert to float in fit

# Target for classification
y_clf = np.array([0, 1, 0, 1, 0, 0, 1, 1, 0, 1])

# Target for regression
y_reg = np.array([2.5, 3.0, 2.7, 3.5, 2.0, 2.8, 3.8, 3.1, 2.6, 4.0])

# Specify categorical feature indices
categorical_features = [1, 3]

# --- Classification Example ---
print("--- Classification ---")
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X, y_clf, test_size=0.3, random_state=42)

# n_children_per_node=2 will split by mean (binary-like behavior)
# n_children_per_node=3 will try to create 3 children per split
classifier = SimilarityDecisionTree_D18(
    isClassifier=True,
    categoricalFeatures=categorical_features,
    maxDepth=3,
    minSamplesLeaf=1,
    n_children_per_node=2, # Try 2 or 3
    weights=[0.25, 0.25, 0.3, 0.2] # Optional custom weights
)
classifier.fit(X_train_clf.astype(float), y_train_clf) # Ensure X is float for processing
predictions_clf = classifier.predict(X_test_clf.astype(float))

print("Test Predictions (Classifier):", predictions_clf)
print("Actual Labels (Classifier):", y_test_clf)
print("Accuracy:", accuracy_score(y_test_clf, predictions_clf))
print("\n")


# --- Regression Example ---
print("--- Regression ---")
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.3, random_state=42)

regressor = SimilarityDecisionTree_D18(
    isClassifier=False,
    categoricalFeatures=categorical_features,
    maxDepth=3,
    minSamplesLeaf=1,
    n_children_per_node=2
)
regressor.fit(X_train_reg.astype(float), y_train_reg)
predictions_reg = regressor.predict(X_test_reg.astype(float))

print("Test Predictions (Regressor):", predictions_reg)
print("Actual Values (Regressor):", y_test_reg)
print("Mean Squared Error:", mean_squared_error(y_test_reg, predictions_reg))
```
## Parameters of `SimilarityDecisionTree_D18`

*   `isClassifier` (bool, default: `True`): If `True`, performs classification. Otherwise, performs regression.
*   `categoricalFeatures` (list of int, default: `None`): A list of indices corresponding to categorical features in the input data `X`.
*   `maxDepth` (int, default: `4`): The maximum depth of the tree.
*   `nJobs` (int, default: `-1`): The number of jobs to run in parallel for tree building. `-1` means using all available processors. `1` means no parallelization.
*   `parallelizationThreshold` (int, default: `500000`): The minimum number of computations (samples * features at a node) to trigger parallel processing for building child nodes.
*   `minSamplesLeaf` (int, default: `1`): The minimum number of samples required to be at a leaf node. A split will only be considered if it leaves at least `minSamplesLeaf` training samples in each of the resulting children.
*   `weights` (list or np.array, default: `None`): A list or array of weights for each feature. If `None`, all features are weighted equally. Weights are normalized internally to sum to 1.
*   `n_children_per_node` (int, default: `2`): The target number of children for each internal node split. The tree uses an iterative mean-splitting approach to try and achieve this many branches.

## How It Works Internally

1.  **Initialization:**
    *   Stores parameters.
    *   Determines a boolean mask for categorical features.
    *   Normalizes provided feature `weights` (or initializes them equally).

2.  **Training (`fit` method):**
    *   **Normalization:** Numeric features in the training data `X` are min-max scaled to the [0,1] range. Parameters for this scaling (`minNumericFeatures`, `numericFeaturesRanges`) are stored for use during prediction. NaNs are handled by `np.nanmin`/`np.nanmax`.
    *   **Tree Building (`build_tree` recursively):**
        *   **Stopping Conditions:** Recursion stops if:
            *   `maxDepth` is reached.
            *   The node is "pure" (all samples have the same class, for classifiers).
            *   The number of samples in the node is less than `minSamplesLeaf` or too few to form `MIN_BRANCHES_FOR_SPLIT` (currently 2) children each with `minSamplesLeaf`.
            *   The range of similarities to the prototype is below `SIMILARITY_EPSILON`.
        *   **Prototype Selection:** A sample is randomly chosen from the current node's data to act as the "prototype."
        *   **Similarity Calculation:** Gower similarity is computed between the prototype and all other samples in the node (see "Gower Similarity and NaN Handling" below).
        *   **Iterative Mean Splitting:**
            *   The algorithm attempts to create `n_children_per_node` child branches.
            *   It starts with all samples in one group.
            *   It iteratively picks the largest available group, calculates the mean of its similarity scores, and attempts to split this group into two using this mean as a threshold (samples <= mean, samples > mean).
            *   This process continues, adding valid split thresholds to a list, until the target number of groups (children) is approached or no more valid splits can be made (due to `minSamplesLeaf` or low similarity variance).
            *   The unique, sorted thresholds found are then used with `np.digitize` to assign all samples in the current node to the final child bins.
        *   **Child Node Creation:** For each resulting child bin with enough samples:
            *   If eligible for parallelization (`parallelizationThreshold` and `nJobs != 1`), child subtrees are built in parallel using `joblib`.
            *   Otherwise, child subtrees are built sequentially.
        *   **Node Storage:** Internal nodes store the `prototype`, the `threshold_edges` used for `np.digitize`, and a list of `children` (subtrees or leaf values). Leaf nodes store the majority class (classifier) or mean value (regressor).

3.  **Prediction (`predict` method):**
    *   Input data `X` is processed:
        *   Numeric features are normalized using the `minNumericFeatures` and `numericFeaturesRanges` stored from training.
    *   Each sample traverses the tree (`_traverse_tree`):
        *   At each internal node, Gower similarity between the sample and the node's `prototype` is calculated.
        *   The sample is directed to a child branch based on its similarity and the node's `threshold_edges` (using `np.digitize`).
        *   This continues until a leaf node is reached, and the leaf's value is assigned as the prediction.

4.  **Gower Similarity and NaN Handling (`gower_similarity_to_prototype`):**
    *   For a given sample and a prototype:
        *   **Numerical Features:** Similarity is `1 - |normalized_value_sample - normalized_value_prototype|`. The range of this per-feature similarity is [0,1].
        *   **Categorical Features:** Similarity is `1` if values are equal, `0` otherwise.
        *   **NaNs:** If a feature value is `NaN` in *either* the sample or the prototype, that specific feature is *ignored* for this sample-prototype comparison.
        *   **Overall Similarity:** The final similarity is a weighted sum of the per-feature similarities, divided by the sum of weights of *only those features that were mutually present (not NaN) and thus actually compared*. If no features can be compared (e.g., due to complementary NaNs), the similarity is 0.

## Potential Limitations & Future Work

*   **Prototype Selection:** Currently, the prototype is chosen randomly. More sophisticated selection methods (e.g., medoids) might improve performance.
*   **Splitting Criterion:** The current n-ary split uses iterative mean-based splitting. Exploring other methods for deriving multiple thresholds (e.g., based on quantiles of similarity, or optimizing for a multi-way split criterion) could be beneficial.
*   **Pruning:** The tree currently has no explicit pruning mechanism, which might lead to overfitting on some datasets.
*   **Computational Cost:** Gower similarity calculation for all pairs (or sample-to-prototype) can be more intensive than simpler attribute tests, especially for many features or large `n_children_per_node`.

## License

Specify your license here (e.g., MIT License, Apache 2.0, etc.). If unsure, MIT is a common permissive license.