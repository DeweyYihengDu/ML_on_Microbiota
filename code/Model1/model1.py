# Load the data, preprocess it, train and evaluate the models


import pandas as pd
import numpy as np
from sklearn.linear_model import Lars, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
import scipy.stats as stats

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    return df

def map_drug_concentration(df):
    drug_columns = ['amoxicillin', 'oxytetracycline_dihydrate', 'sulfadiazine', 'trimethoprim', 'tylosin_tartrate', 'ciprofloxacin']
    for drug in drug_columns:
        df[drug] = df[drug].apply(lambda value: "none" if value == 0 else "low" if value in [5, 10] else "high" if value in [50, 100] else "unknown")
    df['Drug Set'] = df[drug_columns].apply(lambda row: '_'.join(row), axis=1)
    return df

def filter_data(df, bacterial_families):
    df_filtered = df[['SampleID', 'Group', 'Isolation_source', 'Drug Set'] + bacterial_families].dropna()
    return df_filtered

def encode_and_scale_features(df_filtered):
    for col in ['Isolation_source', 'Drug Set']:
        df_filtered.loc[:, col] = LabelEncoder().fit_transform(df_filtered[col])
    X = df_filtered[['Isolation_source', 'Drug Set']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, df_filtered

def train_and_evaluate_models(X_scaled, df_filtered, bacterial_families):
    models = {
        'Least Angle Regression': Lars(),
        'Random Forest': RandomForestRegressor(),
        'Lasso': Lasso(),
        'Elastic-Net': ElasticNet()
    }

    evaluation_metrics = []
    for family in bacterial_families:
        y = df_filtered[family].values
        if np.isnan(y).any():
            print(f"Skipping {family} due to NaN values in target variable.")
            continue

        for model_name, model in models.items():
            loo = LeaveOneOut()
            rmse_scores = np.sqrt(-cross_val_score(model, X_scaled, y, cv=loo, scoring='neg_mean_squared_error'))
            rmse_mean = np.mean(rmse_scores)
            rmse_std_error = stats.sem(rmse_scores)

            evaluation_metrics.append({
                'Family': family,
                'Model': model_name,
                'Train RMSE (mean)': rmse_mean,
                'Train RMSE (SE)': rmse_std_error,
            })

    return pd.DataFrame(evaluation_metrics)

def main():
    file_path = 'matrix/otu_merged_data.csv'  # Update with your file path
    bacterial_families = [
        "o__Bacillales;", "o__Lactobacillales;", "o__Enterobacteriales;",
        "o__Burkholderiales;", "o__Actinomycetales;", "o__Aeromonadales;",
        "o__Pseudomonadales;"
    ]
    
    df = load_and_preprocess_data(file_path)
    df = map_drug_concentration(df)
    df_filtered = filter_data(df, bacterial_families)
    X_scaled, df_filtered = encode_and_scale_features(df_filtered)
    evaluation_metrics = train_and_evaluate_models(X_scaled, df_filtered, bacterial_families)
    
    print(evaluation_metrics.to_string(index=False))

if __name__ == "__main__":
    main()
