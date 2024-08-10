import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Iris dataset
def load_data():
    iris = load_iris()
    data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    data['species'] = iris.target
    return data, iris.target_names

# Preprocess the dataset: split and scale
def preprocess_data(data):
    X = data.drop(columns='species')
    y = data['species']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

# Apply PCA to reduce dimensionality for visualization
def apply_pca(X_train, X_test):
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_test_pca

# Train a Random Forest model and evaluate performance
def train_and_evaluate(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=3)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

# Visualize the PCA results
def plot_pca(X_test_pca, y_test, target_names):
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=X_test_pca[:, 0], y=X_test_pca[:, 1], hue=y_test, palette='viridis', alpha=0.7, s=100)
    plt.title('PCA of Iris Dataset')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Species', labels=target_names)
    plt.grid(True)
    plt.show()

# Main function to execute the workflow
def main():
    data, target_names = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(data)
    X_train_pca, X_test_pca = apply_pca(X_train, X_test)
    train_and_evaluate(X_train_pca, X_test_pca, y_train, y_test)
    plot_pca(X_test_pca, y_test, target_names)

if __name__ == "__main__":
    main()
