# Select the best number of features for the model2


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import RandomOverSampler
import warnings

warnings.filterwarnings('ignore')

def load_and_preprocess_data(filepath):
    """Load data and preprocess"""
    df = pd.read_csv(filepath)

    def map_concentration(value):
        if pd.isna(value):
            return "unknown"
        elif value == 0:
            return "none"
        elif value in [5, 10]:
            return "low"
        elif value in [50, 100]:
            return "high"
        else:
            return "unknown"

    antibiotics = ['amoxicillin', 'oxytetracycline_dihydrate', 'sulfadiazine', 'trimethoprim', 'tylosin_tartrate', 'ciprofloxacin']
    for antibiotic in antibiotics:
        df[antibiotic] = df[antibiotic].apply(map_concentration)

    df['antibiotic_combination'] = df[antibiotics].apply(lambda row: '_'.join(row), axis=1)
    df_cleaned = df.dropna()
    return df_cleaned

def feature_engineering(df_cleaned):
    """Feature engineering"""
    feature_columns = [col for col in df_cleaned.columns if col.startswith('o__')]
    X = df_cleaned[feature_columns]
    X = pd.concat([X, df_cleaned[['Isolation_source', 'Group']]], axis=1)
    X = pd.get_dummies(X, columns=['Isolation_source', 'Group'])

    y = df_cleaned['antibiotic_combination']
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    return X, y_encoded

def select_features(X, y, k):
    """Feature selection"""
    selector = SelectKBest(chi2, k=k)
    X_selected = selector.fit_transform(X, y)
    return X_selected

def evaluate_features(X, y):
    """Evaluate model performance with different numbers of features"""
    accuracy_scores = {}
    for k in range(10, X.shape[1] + 1, 10):
        X_selected = select_features(X, y, k)
        X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = RandomForestClassifier(random_state=42)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        accuracy_scores[k] = acc
    
    return accuracy_scores

def plot_accuracy_scores(accuracy_scores):
    """Plot the relationship between the number of features and accuracy"""
    plt.figure(figsize=(12, 6))
    max_value = max(accuracy_scores.values())
    max_key = [k for k, v in accuracy_scores.items() if v == max_value][0]

    for key, value in accuracy_scores.items():
        color = 'lightblue' if key != max_key else 'black'
        plt.bar(key, value, color=color, width=5)
    
    plt.axhline(y=1/7, color='red', linestyle='dashed', linewidth=2)
    plt.text(2, 1/7-0.01, r'$\frac{1}{7}$', verticalalignment='bottom', horizontalalignment='right', color='red', fontsize=16)
    plt.title('Best Accuracy by Number of Features')
    plt.xlabel('Number of Top Features')
    plt.ylabel('Accuracy')
    plt.show()
    
    return max_key

def main():
    """mian function"""
    filepath = 'matrix/otu_merged_data.csv' 
    df_cleaned = load_and_preprocess_data(filepath)
    X, y_encoded = feature_engineering(df_cleaned)
    accuracy_scores = evaluate_features(X, y_encoded)
    best_k = plot_accuracy_scores(accuracy_scores)
    print(f"The best number of features is: {best_k}")

if __name__ == "__main__":
    main()
