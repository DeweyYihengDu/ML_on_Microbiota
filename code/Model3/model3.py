# Import necessary libraries
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, LeaveOneOut
from sklearn.metrics import mean_squared_error
import numpy as np

# Function to calculate Coefficient of Variance (CV)
def coefficient_of_variance(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mean_actual = np.mean(y_true)
    cv = rmse / mean_actual if mean_actual != 0 else float('inf')
    return cv


def G2_to_G4(df_combined):
# Data preprocessing
    df_grouped_combined = df_combined.groupby(['PlotID', 'Generation', 'File']).mean().reset_index()
    df_grouped_G2 = df_grouped_combined[df_grouped_combined['Generation'] == 'G2']
    df_grouped_G4 = df_grouped_combined[df_grouped_combined['Generation'] == 'G4']

    # Calculate the top 5 bacteria in G4
    mean_abundance_G4 = df_grouped_G4.iloc[:, 3:].mean() # Excluding 'PlotID', 'Generation', 'File'
    top_5_bacteria_G4 = mean_abundance_G4.sort_values(ascending=False).head(5).index.tolist()

    # Prepare the data for training and testing
    X_G2 = pd.get_dummies(df_grouped_G2.drop(columns=['Generation']), columns=['File']).drop(columns=['PlotID'])
    y_G4 = df_grouped_G4[df_grouped_G4['PlotID'].isin(df_grouped_G2['PlotID'])][top_5_bacteria_G4]

    # Model training
    X_train_G2, X_test_G2, y_train_G4, y_test_G4 = train_test_split(X_G2, y_G4, test_size=0.2)
    model_G2_G4 = RandomForestRegressor(random_state=42,n_estimators=2)
    model_G2_G4.fit(X_train_G2, y_train_G4)
    y_pred_G2_G4 = model_G2_G4.predict(X_test_G2)

    # Calculate SE
    se_G2_G4 = np.std(y_test_G4.to_numpy().flatten() - y_pred_G2_G4.flatten())

    # Initialize LeaveOneOut cross-validator
    loo = LeaveOneOut()

    # Perform Leave-One-Out Cross-Validation for CV
    cv_scores = []
    for train_index, test_index in loo.split(X_G2):
        X_train_loo, X_test_loo = X_G2.iloc[train_index], X_G2.iloc[test_index]
        y_train_loo, y_test_loo = y_G4.iloc[train_index], y_G4.iloc[test_index]
        model_G2_G4.fit(X_train_loo, y_train_loo)
        y_pred_loo = model_G2_G4.predict(X_test_loo)
        cv_score = coefficient_of_variance(y_test_loo.to_numpy().flatten(), y_pred_loo.flatten())
        cv_scores.append(cv_score)

    return cv_scores

def G3_to_G4(df_combined):

    # Data preprocessing
    df_grouped_combined = df_combined.groupby(['PlotID', 'Generation', 'File']).mean().reset_index()
    df_grouped_G3 = df_grouped_combined[df_grouped_combined['Generation'] == 'G3']
    df_grouped_G4 = df_grouped_combined[df_grouped_combined['Generation'] == 'G4']

    # Calculate the top 5 bacteria in G4
    mean_abundance_G4 = df_grouped_G4.iloc[:, 3:].mean() # Excluding 'PlotID', 'Generation', 'File'
    top_5_bacteria_G4 = mean_abundance_G4.sort_values(ascending=False).head(5).index.tolist()

    # Prepare the data for training and testing
    X_G3 = pd.get_dummies(df_grouped_G3.drop(columns=['Generation']), columns=['File']).drop(columns=['PlotID'])
    y_G4 = df_grouped_G4[df_grouped_G4['PlotID'].isin(df_grouped_G3['PlotID'])][top_5_bacteria_G4]

    # Model training
    X_train_G3, X_test_G3, y_train_G4, y_test_G4 = train_test_split(X_G3, y_G4, test_size=0.2, random_state=42)
    model_G3_G4 = RandomForestRegressor(random_state=42,n_estimators=5)
    model_G3_G4.fit(X_train_G3, y_train_G4)
    y_pred_G3_G4 = model_G3_G4.predict(X_test_G3)

    # Calculate SE
    se_G3_G4 = np.std(y_test_G4.to_numpy().flatten() - y_pred_G3_G4.flatten())

    # Initialize LeaveOneOut cross-validator
    loo = LeaveOneOut()

    # Perform Leave-One-Out Cross-Validation for CV
    cv_scores = []
    for train_index, test_index in loo.split(X_G3):
        X_train_loo, X_test_loo = X_G3.iloc[train_index], X_G3.iloc[test_index]
        y_train_loo, y_test_loo = y_G4.iloc[train_index], y_G4.iloc[test_index]
        model_G3_G4.fit(X_train_loo, y_train_loo)
        y_pred_loo = model_G3_G4.predict(X_test_loo)
        cv_score = coefficient_of_variance(y_test_loo.to_numpy().flatten(), y_pred_loo.flatten())
        cv_scores.append(cv_score)

    # Calculate mean CV score
    mean_cv_score = np.mean(cv_scores)

    return cv_scores



def G23_to_G4_combined(df_combined):
    # Data preprocessing
    df_grouped_combined = df_combined.groupby(['PlotID', 'Generation', 'File']).mean().reset_index()
    df_grouped_G2_combined = df_grouped_combined[df_grouped_combined['Generation'] == 'G2'].drop(columns=['Generation'])
    df_grouped_G3_combined = df_grouped_combined[df_grouped_combined['Generation'] == 'G3'].drop(columns=['Generation'])
    df_grouped_G4_combined = df_grouped_combined[df_grouped_combined['Generation'] == 'G4'].drop(columns=['Generation'])
    df_grouped_G2_combined.columns = [str(col) + '_G2' for col in df_grouped_G2_combined.columns]
    df_grouped_G3_combined.columns = [str(col) + '_G3' for col in df_grouped_G3_combined.columns]
    df_grouped_G2_G3_combined = pd.merge(df_grouped_G2_combined.rename(columns={'PlotID_G2': 'PlotID', 'File_G2': 'File'}),
                                        df_grouped_G3_combined.rename(columns={'PlotID_G3': 'PlotID', 'File_G3': 'File'}),
                                        on=['PlotID', 'File'])

    # Calculate the top 5 bacteria in G4
    mean_abundance_grouped_G4_combined = df_grouped_G4_combined.iloc[:, 1:].mean()
    top_5_bacteria_grouped_G4_combined = mean_abundance_grouped_G4_combined.sort_values(ascending=False).head(5).index.tolist()

    # Prepare the data for training and testing
    X_grouped_G2_G3_combined = pd.get_dummies(df_grouped_G2_G3_combined, columns=['File']).drop(columns=['PlotID'])
    y_grouped_G4_combined = df_grouped_G4_combined[df_grouped_G4_combined['PlotID'].isin(df_grouped_G2_G3_combined['PlotID'])][top_5_bacteria_grouped_G4_combined]

    # Model training
    X_train_grouped_G2_G3_combined, X_test_grouped_G2_G3_combined, y_train_grouped_G4_combined, y_test_grouped_G4_combined = train_test_split(X_grouped_G2_G3_combined, y_grouped_G4_combined, test_size=0.2, random_state=42)
    model_grouped_G2_G3_G4_combined = RandomForestRegressor()
    model_grouped_G2_G3_G4_combined.fit(X_train_grouped_G2_G3_combined, y_train_grouped_G4_combined)
    y_pred_grouped_G2_G3_G4_combined = model_grouped_G2_G3_G4_combined.predict(X_test_grouped_G2_G3_combined)

    # Calculate SE
    se_grouped_G2_G3_G4_combined = np.std(y_test_grouped_G4_combined.to_numpy().flatten() - y_pred_grouped_G2_G3_G4_combined.flatten())

    # Initialize LeaveOneOut cross-validator
    loo = LeaveOneOut()

    # Perform Leave-One-Out Cross-Validation for CV
    cv_scores = []
    for train_index, test_index in loo.split(X_grouped_G2_G3_combined):
        X_train_loo, X_test_loo = X_grouped_G2_G3_combined.iloc[train_index], X_grouped_G2_G3_combined.iloc[test_index]
        y_train_loo, y_test_loo = y_grouped_G4_combined.iloc[train_index], y_grouped_G4_combined.iloc[test_index]
        model_grouped_G2_G3_G4_combined.fit(X_train_loo, y_train_loo)
        y_pred_loo = model_grouped_G2_G3_G4_combined.predict(X_test_loo)
        cv_score = coefficient_of_variance(y_test_loo.to_numpy().flatten(), y_pred_loo.flatten())
        cv_scores.append(cv_score)

    # Calculate mean CV score
    mean_cv_score = np.mean(cv_scores)

    return cv_scores
def main():
    files = [
        'standardized_high_high_high_high_high_high.csv',
        'standardized_high_high_high_none_none_none.csv',
        'standardized_high_none_none_none_none_none.csv',
        'standardized_low_low_low_low_low_low.csv',
        'standardized_low_low_low_none_none_none.csv',
        'standardized_low_none_none_none_none_none.csv',
        'standardized_none_none_none_none_none_none.csv'
    ]
    df_list = []
    for file in files:
        df_temp = pd.read_csv(f'matrix/set_plot/{file}')
        df_temp['File'] = file
        df_list.append(df_temp)
    df_combined = pd.concat(df_list, ignore_index=True)
    a23 = G23_to_G4_combined(df_combined)
    a3 = G3_to_G4(df_combined)
    a2 = G2_to_G4(df_combined)
    df = pd.DataFrame({
    'G2 to G3': a2,
    'G3 to G4': a3,
    'G2 & G3 to G4': a23})
    print(df.to_string(index=False))

if __name__ == '__main__':
    main()
