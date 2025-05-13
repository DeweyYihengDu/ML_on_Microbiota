import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import RandomOverSampler
import warnings

warnings.filterwarnings('ignore')


df = pd.read_csv('../../matrix/otu_merged_data.csv')

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

def map_to_set(row):
    if all(value == 'unknown' for value in row):
        return 'Unknown'
    mapping = {
        'high_high_high_high_high_high': 'Set 1',
        'high_high_high_none_none_none': 'Set 2',
        'high_none_none_none_none_none': 'Set 3',
        'low_low_low_low_low_low': 'Set 4',
        'low_low_low_none_none_none': 'Set 5',
        'low_none_none_none_none_none': 'Set 6',
        'none_none_none_none_none_none': 'Control'
    }
    key = '_'.join(row)
    return mapping.get(key, 'Other')

df['set_name'] = df[antibiotics].apply(map_to_set, axis=1)

df_cleaned = df.dropna()

feature_columns = [col for col in df_cleaned.columns if col.startswith('o__')]
X = df_cleaned[feature_columns]
X = pd.concat([X, df_cleaned[['Isolation_source', 'Group']]], axis=1)
X = pd.get_dummies(X, columns=['Isolation_source', 'Group'])

y = df_cleaned['set_name']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

best_k = 20 # get from previous example
selector = SelectKBest(chi2, k=best_k)
X_selected = selector.fit_transform(X, y_encoded)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

ros = RandomOverSampler()
X_resampled, y_resampled = ros.fit_resample(X_scaled, y_encoded)
X_balanced, y_balanced = X_resampled, y_resampled

models = {
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(probability=True),
    "Decision Tree": DecisionTreeClassifier(),
    "Logistic Regression": LogisticRegression()
}

param_grids = {
    "Random Forest": {'n_estimators': [100, 200], 'max_depth': [10, 20]},
    "SVM": {'C': [1, 10], 'kernel': ['rbf', 'linear']},
    "Decision Tree": {'max_depth': [5, 10]},
    "Logistic Regression": {'C': [1, 10]}
}

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 12))
axes = axes.flatten() 

for ax, (name, model) in zip(axes, models.items()):
    grid_search = GridSearchCV(model, param_grids[name], cv=5, scoring='roc_auc_ovr')
    grid_search.fit(X_balanced, y_balanced)
    best_model = grid_search.best_estimator_
    
    y_pred = cross_val_predict(best_model, X_balanced, y_balanced, cv=5)
    y_balanced_labels = label_encoder.inverse_transform(y_balanced)
    y_pred_labels = label_encoder.inverse_transform(y_pred)
    cm = confusion_matrix(y_balanced_labels, y_pred_labels, labels=label_encoder.classes_)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    annot_array = np.vectorize(lambda x, y: f'{x}\n({y:.2%})')(cm, cm_normalized)
    
    sns.heatmap(cm_normalized, annot=annot_array, fmt="", cmap='Blues', ax=ax)
    ax.set_title(f'Normalized Confusion Matrix for {name}', fontsize=14)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_xticklabels(labels=label_encoder.classes_, rotation=45)
    ax.set_yticklabels(labels=label_encoder.classes_, rotation=45)

plt.tight_layout()

# plt.savefig('confusion_matrices.pdf', format='pdf', dpi=300)