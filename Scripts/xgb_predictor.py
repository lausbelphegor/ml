# Model and performance evaluation
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_fscore_support as score

# Standardize the data
from sklearn.preprocessing import StandardScaler

# Hyperparameter tuning
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from hyperopt import tpe, STATUS_OK, Trials, hp, fmin, STATUS_OK, space_eval

# Data processing
import pandas as pd
import numpy as np

# Plot the findings
import matplotlib.pyplot as plt
import plotly.express as px


# TODO MACD
# TODO RSI
# TODO Bollinger Bands
# TODO Stochastic Oscillator
# TODO Stochastic RSI


# Load your dataset
data = pd.read_csv("Scripts\Data\BTC-USD.csv")
sentiment = pd.read_csv("Scripts\Data\sentiment_means.csv")

TEST_SIZE = 100
GAP_BETWEEN_TRAIN_AND_TEST = 0
SPLITS = 15
max_splits = (len(
    data) - GAP_BETWEEN_TRAIN_AND_TEST) // (TEST_SIZE + GAP_BETWEEN_TRAIN_AND_TEST)
FEATURES = ['Close_Diff', 'Sentiment_Mean']
TARGET = 'Target'
SPLITS = min(SPLITS, max_splits)

param_grid = {
    'n_estimators': [50, 100],
    'learning_rate': [0.1, 0.2, 0.3],
    'max_depth': [3, 4, 5],
    'subsample': [0.6, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 1.0],
    'colsample_bylevel': [0.6,  0.9, 1.0],
    'min_child_weight': [1, 2,  20],
    'gamma': [0, 0.1,  0.4, 0.5],
    'scale_pos_weight': [1, 2, 3, 4],
    'reg_alpha': [0, 0.1, 0.5, 1],
    'reg_lambda': [1, 1.5, 2, 3, 4.5],
    'booster': ['gbtree']
}
# param_grid = {
#     'n_estimators': [50, 100, 500, 1000, 1500],
#     'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2, 0.3],
#     'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
#     'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
#     'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
#     'colsample_bylevel': [0.6, 0.7, 0.8, 0.9, 1.0],
#     'min_child_weight': [1, 2, 5, 10, 20],
#     'gamma': [0, 0.1, 0.2, 0.3, 0.4, 0.5],
#     'scale_pos_weight': [1, 2, 3, 4],
#     'reg_alpha': [0, 0.1, 0.5, 1],
#     'reg_lambda': [1, 1.5, 2, 3, 4.5],
#     'booster': ['gbtree']
# }


data.drop(['Open', 'High', 'Low', 'Adj Close', 'Volume'], axis=1, inplace=True)

data['Target'] = data['Close'].shift(periods=-1)
data['Target_tmp'] = data['Target']
data['Target'].mask(data['Target_tmp'] >= data['Close'], 0, inplace=True)
data['Target'].mask(data['Target_tmp'] < data['Close'], 1, inplace=True)

data.drop(['Target_tmp'], axis=1, inplace=True)

data['Close_Diff'] = data['Close'].shift(periods=1)
data['Close_Diff'] = data['Close'] - data['Close_Diff']

data.dropna(inplace=True)

data['Date'] = pd.to_datetime(data['Date'])
sentiment['Date'] = pd.to_datetime(sentiment['Date'])
data.set_index('Date', inplace=True)

data = data.merge(sentiment, on='Date', how='left')

# Check the target value distribution
print(data['Target'].value_counts(normalize=True))
# ---------------------------------------------

tss = TimeSeriesSplit(n_splits=SPLITS,
                      test_size=TEST_SIZE,
                      gap=GAP_BETWEEN_TRAIN_AND_TEST)

# fig, axs = plt.subplots(SPLITS, 1, figsize=(15, 15), sharex=True)

# fold = 0
# for train_idx, val_idx in tss.split(data):
#     train = data.iloc[train_idx]
#     test = data.iloc[val_idx]
#     train['Close_Diff'].plot(ax=axs[fold],
#                              label='Training Set',
#                              title=f'Data Train/Test Split Fold {fold}')
#     test['Close_Diff'].plot(ax=axs[fold],
#                             label='Test Set')
#     axs[fold].axvline(test.index.min(), color='black', ls='--')
#     fold += 1

# plt.show()


# ---------------------------------------------

# Initialize the model
model = XGBClassifier(
    objective='binary:logistic',
    tree_method='auto',
    max_bin=256,
    seed=0
)

# Create the random search object
random_search = RandomizedSearchCV(model, param_distributions=param_grid,
                                   n_iter=50,
                                   scoring='roc_auc',  # Using AUC-ROC as scoring metric
                                   n_jobs=-1,
                                   cv=tss,
                                   verbose=1)


def train_model():
    fold = 0
    preds = []
    scores = []
    for train_idx, val_idx in tss.split(data):

        # Initiate scaler
        sc = StandardScaler()

        # Standardize the training dataset
        X_train_transformed = pd.DataFrame(sc.fit_transform(
            X_train), index=X_train.index, columns=X_train.columns)

        # Standardized the testing dataset
        X_test_transformed = pd.DataFrame(sc.transform(
            X_test), index=X_test.index, columns=X_test.columns)

        train = data.iloc[train_idx]
        test = data.iloc[val_idx]

        X_train = train[FEATURES]
        y_train = train[TARGET]

        X_test = test[FEATURES]
        y_test = test[TARGET]

        # Fitting the random search to each fold
        random_search.fit(X_train, y_train)

        # Getting the best estimator from the random search
        best_model = random_search.best_estimator_

        y_pred = best_model.predict(X_test)
        preds.append(y_pred)

        score = np.sqrt(mean_square_error(y_test, y_pred))
        scores.append(score)

        fold += 1

        print(f"Best Parameters for fold {fold}: ", random_search.best_params_)

    print(f'Score across folds {np.mean(scores):0.4f}')
    print(f'Fold scores: {scores}')


train_model()

# ---------------------------------------------

# Convert results to DataFrame
results = pd.DataFrame(random_search.cv_results_)

# Extract the parameters and the scores
selected_columns = [
    col for col in results.columns if "param_" in col] + ['mean_test_score']
df = results[selected_columns]

# Rename 'mean_test_score' for clarity in plot
df = df.rename(columns={'mean_test_score': 'Score'})

# Create parallel coordinates plot
fig = px.parallel_coordinates(df, color="Score", labels={'Score': 'Mean Test Score'},
                              color_continuous_scale=px.colors.diverging.Tealrose,
                              color_continuous_midpoint=df.Score.median())

fig.show()

# ---------------------------------------------


# print(data.head())
# print(data.tail())
