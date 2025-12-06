# Classification

This folder contains comprehensive tutorials on **classification algorithms** in supervised learning, covering logistic regression, decision trees, and the mathematical foundations behind tree-based splitting criteria.

## 📚 Contents

### 1. `Classification.ipynb`
**Classification Algorithms: Logistic Regression, Decision Trees, and Random Forest**

This notebook provides hands-on experience with multiple classification algorithms:

- **Supervised Learning Recap**:
  - Classification vs Regression
  - Working with categorical labels

- **Logistic Regression (Multiclass)**:
  - Mathematical foundation:
    - Binary: Sigmoid function $\sigma(z) = \frac{1}{1 + e^{-z}}$
    - Multiclass: Softmax function for $K$ classes
  - Implementation on Iris dataset:
    - Feature scaling with StandardScaler
    - Pipeline creation (scaler + logistic regression)
    - Stratified train/test split
    - Classification report interpretation
    - Predicted probabilities analysis

- **Decision Trees**:
  - Splitting criteria explained:
    - **Entropy**: $H(t) = -\sum_{i=1}^{C} p_i \log_2(p_i)$
    - **Information Gain**: $IG = H(\text{parent}) - \sum_{k} \frac{N_k}{N} H(\text{child}_k)$
    - **Gini Impurity**: $G(t) = 1 - \sum_{i=1}^{C} p_i^2$
  - Implementation:
    - Decision tree classifier with max_depth=3
    - Feature importance interpretation
    - Accuracy evaluation

- **Random Forest**:
  - Ensemble of decision trees
  - Feature importance aggregation
  - Comparison with single decision tree

**Key Learning Outcomes:**
- Understand multiclass classification with softmax
- Learn decision tree splitting criteria (entropy, Gini)
- Compare different classification algorithms
- Interpret feature importances
- Understand when to use each algorithm

### 2. `InformationGain.ipynb`
**Deep Dive into Decision Tree Learning: Entropy and Information Gain**

A step-by-step mathematical exploration of how decision trees learn to split data:

- **The Big Idea**:
  - Decision trees as flowcharts of if-then questions
  - Goal: Create pure groups (single class per node)

- **Step-by-Step Learning Process**:
  1. Start with mixed parent node
  2. Measure impurity (entropy or Gini)
  3. Try possible splits
  4. Calculate information gain
  5. Choose best split
  6. Repeat recursively

- **Mathematical Foundations**:
  - **Entropy Calculation**: Detailed walkthrough with student pass/fail example
  - **Information Gain**: How much uncertainty is removed by a split
  - **Gini Impurity**: Alternative purity measure
  - Manual calculations with 6-student dataset

- **Practical Example**:
  - Dataset: 6 students with hours studied and pass/fail outcome
  - Manual entropy calculation for parent node
  - Testing split at "Hours Studied < 5"
  - Computing entropy for child nodes
  - Calculating information gain
  - Comparing entropy vs Gini impurity

**Key Learning Outcomes:**
- Understand the mathematical intuition behind decision trees
- Learn to calculate entropy and information gain by hand
- See how trees choose optimal splits
- Understand the relationship between entropy and Gini impurity
- Gain deep intuition for tree-based algorithms

## 🛠️ Technologies Used

- **pandas**: Data manipulation
- **numpy**: Numerical operations
- **scikit-learn**:
  - `LogisticRegression`: Multiclass classification
  - `DecisionTreeClassifier`: Tree-based classification
  - `RandomForestClassifier`: Ensemble method
  - `StandardScaler`: Feature scaling
  - `Pipeline`: Model pipeline creation
  - `train_test_split`: Data splitting
  - `classification_report`, `accuracy_score`: Evaluation metrics

## 📋 Prerequisites

- Understanding of supervised learning
- Basic knowledge of probability and logarithms
- Familiarity with pandas and scikit-learn
- Understanding of train/test splits

## 🚀 Getting Started

1. **Install Required Packages**:
   ```bash
   pip install pandas numpy scikit-learn matplotlib
   ```

2. **Run the Notebooks**:
   - Start with `Classification.ipynb` for algorithm implementations
   - Follow with `InformationGain.ipynb` for mathematical deep dive

## 📊 Datasets Used

1. **Iris Dataset** (Classification.ipynb):
   - 150 samples, 4 features (sepal length, sepal width, petal length, petal width)
   - 3 classes: Setosa, Versicolor, Virginica
   - Classic multiclass classification problem

2. **Student Pass/Fail Dataset** (InformationGain.ipynb):
   - 6 students with hours studied
   - Binary classification: Pass (1) or Fail (0)
   - Small dataset for manual calculation demonstration

## 💡 Key Concepts

### Entropy
- Measures uncertainty/impurity in a node
- Range: 0 (pure) to 1 (maximum uncertainty for binary)
- Formula uses base-2 logarithm (bits of information)

### Information Gain
- Measures reduction in entropy after a split
- Higher gain = better split
- Formula: Parent entropy - weighted average of child entropies

### Gini Impurity
- Alternative to entropy for measuring impurity
- Range: 0 (pure) to 0.5 (maximum for binary)
- Simpler formula: $1 - \sum p_i^2$
- Often faster to compute than entropy

### Softmax (Multiclass Logistic Regression)
- Generalizes sigmoid to multiple classes
- Ensures probabilities sum to 1
- Formula: $p(y=k|x) = \frac{e^{\theta_k^T x}}{\sum_{j=1}^{K} e^{\theta_j^T x}}$

## 🎯 Algorithm Comparison

| Algorithm | Pros | Cons | Best For |
|-----------|------|------|----------|
| **Logistic Regression** | Interpretable, fast, probabilistic outputs | Linear boundaries only | Linearly separable data |
| **Decision Tree** | Interpretable, handles non-linear patterns | Prone to overfitting | Small datasets, feature importance |
| **Random Forest** | Robust, handles non-linear, feature importance | Less interpretable, slower | Complex patterns, large datasets |

## 📝 Notes

- `InformationGain.ipynb` includes detailed manual calculations
- All formulas are explained with step-by-step examples
- Results are reproducible with fixed random seeds
- Mathematical foundations are emphasized for deep understanding

## 🔗 Related Topics

- **Model Evaluation**: See `model_evaluation/` for metrics and evaluation techniques
- **Ensemble Methods**: See `ensamble/` for advanced ensemble techniques
- **Dimensionality Reduction**: See `dimentionality_reduction/` for feature selection/extraction
