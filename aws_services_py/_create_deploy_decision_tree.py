import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.externals import joblib  # to save the model and transformers

# 1. Load or create your dataset
# Here we'll use a synthetic dataset for illustration.
# You can load your dataset using pandas (e.g., pd.read_csv()).
X = pd.DataFrame({
    'feature1': [1, 2, 3, 4, 5],
    'feature2': ['A', 'B', 'A', 'B', 'A'],
    'feature3': [10, 20, 30, 40, 50]
})
y = np.array([0, 1, 0, 1, 0])  # Binary classification target

# 2. Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Define preprocessing steps (you can add more if needed)
# Example: scaling numerical features and one-hot encoding categorical features.
numerical_features = ['feature1', 'feature3']
categorical_features = ['feature2']

# Preprocessing pipeline for numerical features
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Handle missing values
    ('scaler', StandardScaler())  # Scale features
])

# Preprocessing pipeline for categorical features
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),  # Handle missing values
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One-hot encode categorical variables
])

# Combine the transformations for both numerical and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# 4. Define the Decision Tree Classifier
model = DecisionTreeClassifier(random_state=42)

# 5. Create the full pipeline (preprocessing + model)
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', model)
])

# 6. Fit the pipeline on the training data
pipeline.fit(X_train, y_train)

# 7. Save the pipeline (which includes preprocessing and model) for future use
joblib.dump(pipeline, 'decision_tree_model_pipeline.pkl')

# 8. Load the pipeline for future inference
loaded_pipeline = joblib.load('decision_tree_model_pipeline.pkl')

# 9. Use the loaded pipeline to make predictions on new data
predictions = loaded_pipeline.predict(X_test)

print("Predictions:", predictions)

'''
Visualizing a decision tree in a Jupyter notebook is straightforward using sklearn's built-in functions, particularly plot_tree and export_graphviz. Below are different methods to visualize the decision tree, including options for simple plots and more detailed visualizations.

1. Visualizing a Decision Tree with plot_tree
plot_tree from sklearn.tree is a simple and effective way to visualize the structure of a decision tree directly in Jupyter notebooks.

Here is how you can do it:

python
Copy code
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Assuming you already have a trained decision tree model (e.g., 'model')
# For demonstration, we'll visualize a decision tree from a trained pipeline

# Visualize the decision tree
plt.figure(figsize=(12, 8))
plot_tree(model, filled=True, feature_names=['feature1', 'feature2', 'feature3'], class_names=['class_0', 'class_1'], rounded=True)
plt.show()
Parameters in plot_tree:
filled: Colors the nodes according to the majority class or the mean value.
feature_names: List of feature names used in the tree.
class_names: List of class names to display at the leaf nodes.
rounded: Rounds the corners of the nodes for a more polished appearance.
2. Visualizing a Decision Tree with export_graphviz
export_graphviz allows you to export the decision tree to the Graphviz format, which you can then render into a graphical representation.

python
Copy code
from sklearn.tree import export_graphviz
import graphviz

# Export the decision tree as a .dot file
dot_data = export_graphviz(model, out_file=None, 
                           feature_names=['feature1', 'feature2', 'feature3'],
                           class_names=['class_0', 'class_1'],
                           filled=True, rounded=True, special_characters=True)

# Render the .dot file into a graph
graph = graphviz.Source(dot_data)
graph.render('decision_tree_visualization')  # This will save the file as 'decision_tree_visualization.pdf'
graph.view()  # Opens the visualization in the default viewer
How to use the above:
export_graphviz: This function exports the decision tree structure to a dot format file.
out_file=None allows the result to be returned as a string instead of writing it to a file.
feature_names and class_names are used to provide more readable output.
graphviz.Source: Converts the dot string into a renderable graph.
graph.render() saves the tree visualization as a .pdf file or another format.
graph.view() opens the visualization in your default viewer (PDF viewer, for example).
3. Interactive Visualization with dtreeviz (Advanced)
dtreeviz is a Python library for visualizing decision trees in an interactive and detailed manner. It can provide a more comprehensive visualization than plot_tree and export_graphviz.

Installation:
You may need to install dtreeviz first if you haven't:

bash
Copy code
pip install dtreeviz
Usage:
python
Copy code
from dtreeviz.trees import dtreeviz

# Visualize the decision tree with dtreeviz
viz = dtreeviz(model, X_train, y_train, 
               target_name="Target", feature_names=['feature1', 'feature2', 'feature3'],
               class_names=['class_0', 'class_1'])

# Render the tree in a Jupyter notebook cell
viz.view()
dtreeviz offers an interactive, more colorful tree with additional information (like decision thresholds) at each node.

Summary of Methods:
plot_tree: Simple, fast, and integrated with sklearn. Good for quick visualization.
export_graphviz + graphviz: More customizable and professional-looking, but requires external tools (Graphviz).
dtreeviz: Rich, interactive, and visually appealing. Ideal for a detailed visualization with explanations.
Each method has its pros and cons, depending on your requirements for simplicity, interactivity, and aesthetics.




When Decision Trees are Better:
Non-Linear Relationships:

Decision trees work well with non-linear relationships between the features and the target. Logistic regression assumes a linear relationship between the input features and the log-odds of the target, which can be limiting for more complex patterns in the data.
Example: If your data has intricate, non-linear decision boundaries (e.g., XOR-like patterns), decision trees can model this well, whereas logistic regression might struggle.
Categorical Data:

Decision trees handle categorical variables naturally. You don’t need to pre-process or one-hot encode categorical data, as decision trees automatically handle them during the splits.
Logistic Regression requires numerical input and thus often requires preprocessing (like one-hot encoding or label encoding) for categorical variables.
Feature Importance:

Decision trees provide clear insight into the importance of different features in making predictions. They can help you understand which features are most influential in the decision-making process.
Logistic regression also gives feature coefficients, but these are less intuitive when dealing with non-linear relationships or categorical variables.
Interpretability:

Decision trees are easy to interpret and visualize, which can be crucial in applications where model transparency is required (e.g., in regulatory or financial environments). You can follow the path of decisions in a tree to understand the logic behind predictions.
Logistic regression is also interpretable, but as the complexity of the data increases (i.e., more features, non-linearities), the interpretation can become less intuitive, especially when feature interactions exist.
Handling Missing Values:

Decision trees handle missing values fairly well, as they can split the data based on available features and still handle missing entries without imputing them explicitly.
Logistic regression typically requires the data to be clean and doesn’t handle missing values directly (though you could impute missing data before fitting the model).
Complex Interactions:

Decision trees naturally handle complex feature interactions. Trees can split based on multiple features in a non-linear way and model interactions between features without explicitly specifying them.
Logistic regression assumes additive relationships between features and the log-odds of the target, which may miss complex interactions unless manually specified (e.g., using polynomial or interaction terms).
When Logistic Regression is Better:
Linear Relationships:

Logistic regression performs well when there is a linear relationship between the input features and the target. If the decision boundary between the classes is approximately linear, logistic regression will often perform well and provide a simpler model.
Example: If your target is linearly separable (like predicting whether someone will buy a product based on a few key features like age and income), logistic regression may be a better choice.
Smaller or Simpler Data:

Logistic regression tends to perform well on smaller or simpler datasets, especially when there’s a clear linear relationship between features and the outcome.
Decision trees can easily overfit on smaller datasets unless carefully pruned or regularized, so in simpler datasets, logistic regression might be more robust.
Avoiding Overfitting:

Logistic regression is less prone to overfitting than decision trees, especially if the number of features is large relative to the number of observations. Logistic regression models typically have fewer parameters and simpler models, making them less likely to overfit.
Decision trees are more prone to overfitting, especially with deeper trees or when there’s noise in the data. Pruning and regularization techniques (e.g., limiting depth or the minimum number of samples per split) are needed to combat this.
Probabilistic Outputs:

Logistic regression provides probabilistic outputs directly (the predicted probabilities that a sample belongs to a certain class), which can be useful in cases where you want to interpret the certainty of predictions.
Decision trees can also provide probabilities based on the proportion of samples in the leaf nodes, but they might not be as calibrated or accurate as logistic regression in providing well-calibrated probabilities, especially in the case of imbalanced data.
Fast Training:

Logistic regression is computationally faster to train on datasets with many features because it solves a convex optimization problem with a well-defined objective function.
Decision trees may take longer to train, especially when the tree is deep and needs to explore many splits.
Numerical Stability and Regularization:

Logistic regression can be more numerically stable and easier to regularize using techniques like L1 (Lasso) or L2 (Ridge) regularization. Regularization can help mitigate overfitting, especially when you have a lot of features.
Decision trees may need additional mechanisms like pruning or ensemble methods (e.g., Random Forests or Gradient Boosting) to reduce overfitting.
Summary of Key Differences:
Criteria	Decision Tree	Logistic Regression
Model Type	Non-linear (piecewise constant)	Linear (log-odds)
Handling of Categorical Data	Can handle categorical data natively	Requires encoding (e.g., one-hot)
Feature Interactions	Naturally handles interactions	Must specify interactions explicitly
Interpretability	Very interpretable (visualizable)	Interpretable but more challenging with many features
Handling Non-linearity	Can capture non-linear relationships	Assumes linearity
Overfitting	Prone to overfitting, requires pruning	Less prone to overfitting with regularization
Speed	Slower for complex datasets	Faster for simple problems, especially with regularization
Probabilistic Outputs	Can be less accurate; may require calibration	Provides direct probabilities
Missing Values	Handles missing values internally	Requires preprocessing for missing data
Choosing Between Them:
Use a Decision Tree when:

You have complex, non-linear relationships.
You need a model that can handle categorical data and produce easily interpretable decision rules.
You want to capture interactions between features without explicitly modeling them.
Interpretability is key (e.g., in regulated industries).
Use Logistic Regression when:

The problem exhibits a linear relationship between features and the target.
You have a lot of features, and a simple, regularized model might work better for generalization.
You need probabilistic outputs or the model needs to be fast and simple.
Overfitting is a concern, and you want more numerical stability with regularization.
In some cases, it may be beneficial to try both models and use cross-validation to compare performance on your specific dataset.



'''