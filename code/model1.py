import pandas as pd
import numpy as np
from sklearn.linear_model import Lars, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
import scipy.stats as stats

# Function to map drug concentrations to categories
def map_drug_concentration(value):
    """Map numeric drug concentration values to categorical labels."""
    if value == 0:
        return "none"
    elif value in [5, 10]:
        return "low"
    elif value in [50, 100]:
        return "high"
    else:
        return "unknown"

# Load data
file_path = '../matrix/otu_merged_data.csv'
df = pd.read_csv(file_path)

# Create 'Drug Set' column by applying concentration mapping
drug_columns = ['amoxicillin', 'oxytetracycline_dihydrate', 'sulfadiazine', 'trimethoprim', 'tylosin_tartrate', 'ciprofloxacin']
df['Drug Set'] = df[drug_columns].apply(lambda row: '_'.join(map_drug_concentration(x) for x in row), axis=1)

# Filter data to include specific bacterial families
bacterial_families = [
    "o__Bacillales;", "o__Lactobacillales;", "o__Enterobacteriales;", 
    "o__Burkholderiales;", "o__Actinomycetales;", "o__Aeromonadales;", "o__Pseudomonadales;"
]
df_filtered = df[['SampleID', 'Group', 'Isolation_source', 'Drug Set'] + bacterial_families]

# Splitting the data into training and test sets based on 'Group'
train_df = df_filtered[df_filtered['Group'].isin(['G1', 'G2', 'G3'])].copy()
test_df = df_filtered[df_filtered['Group'] == 'G4'].copy()

# Function to encode categorical features
def encode_features(train_df, test_df, columns):
    """Encode categorical features using LabelEncoder."""
    label_encoders = {}
    for col in columns:
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col])
        test_df[col] = le.transform(test_df[col])
        label_encoders[col] = le
    return train_df, test_df, label_encoders

# Encode 'Isolation_source' and 'Drug Set'
train_df, test_df, label_encoders = encode_features(train_df, test_df, ['Isolation_source', 'Drug Set'])

# Define features and target
X_train = train_df[['Isolation_source', 'Drug Set']]
y_train = train_df[bacterial_families]
X_test = test_df[['Isolation_source', 'Drug Set']]
y_test = test_df[bacterial_families]

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train models
models = {
    'Least Angle Regression': Lars(),
    'Random Forest': RandomForestRegressor(),
    'Lasso': Lasso(),
    'Elastic-Net': ElasticNet()
}

# Evaluate models using Leave-One-Out Cross-Validation and store metrics
evaluation_metrics = {}
for family in bacterial_families:
    evaluation_metrics[family] = {}
    for model_name, model in models.items():
        # Cross-validation
        loo = LeaveOneOut()
        rmse_scores = np.sqrt(-cross_val_score(model, X_train_scaled, y_train[family], cv=loo, scoring='neg_mean_squared_error'))
        
        # Summary statistics
        rmse_mean = np.mean(rmse_scores)
        rmse_std_error = stats.sem(rmse_scores)
        
        # Model fitting and evaluation
        model.fit(X_train_scaled, y_train[family])
        y_pred = model.predict(X_test_scaled)
        test_rmse = np.sqrt(mean_squared_error(y_test[family], y_pred))
        
        # Storing metrics
        evaluation_metrics[family][model_name] = {
            'Train RMSE': rmse_mean,
            'Train Standard Error': rmse_std_error,
            'Test RMSE': test_rmse
        }

# Optionally, print or analyze evaluation_metrics here

