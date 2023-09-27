# %%
# Model and performance evaluation
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, make_scorer

# Standardize the data
from sklearn.preprocessing import StandardScaler

# Hyperparameter tuning
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from hyperopt import tpe, STATUS_OK, Trials, hp, fmin, space_eval

# Data processing
import pandas as pd
import numpy as np

# Plot the findings
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
from sklearn.metrics import ConfusionMatrixDisplay
import plotly.express as px
import plotly.figure_factory as ff


# %%
# Load your dataset
data = pd.read_csv(
    r'C:\Users\Archy\Desktop\ml\Project\Scripts\Data\BTC-USD.csv')
sentiment = pd.read_csv(
    r'C:\Users\Archy\Desktop\ml\Project\Scripts\Data\sentiment_means.csv')

# n_samples - gap - (test_size * n_splits) <= 0
TEST_SIZE = 200
GAP_BETWEEN_TRAIN_AND_TEST = 0
SPLITS = 15
FEATURES = ['Close_Diff', 'Sentiment_Mean']
TARGET = 'Target'

# Initiate scaler
sc = StandardScaler()

data.shape

# %%

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


# %%
# Check the target value distribution
data['Target'].value_counts(normalize=True)

# %%
tss = TimeSeriesSplit(n_splits=SPLITS,
                      test_size=TEST_SIZE,
                      gap=GAP_BETWEEN_TRAIN_AND_TEST)

fig, axs = plt.subplots(SPLITS, 1, figsize=(15, 15), sharex=True)

fold = 0
for train_idx, val_idx in tss.split(data):
    train = data.iloc[train_idx]
    test = data.iloc[val_idx]
    train['Close_Diff'].plot(ax=axs[fold],
                             label='Training Set',
                             title=f'Data Train/Test Split Fold {fold}')
    test['Close_Diff'].plot(ax=axs[fold],
                            label='Test Set')
    axs[fold].axvline(test.index.min(), color='black', ls='--')
    fold += 1

plt.show()


# %%
# Initialize the model
model = XGBClassifier(
    objective='binary:logistic',
    tree_method='auto',
    max_bin=256,
    seed=0
)

# %%
# Ref
# param_grid = {
#     'n_estimators': [500, 1000, 1500],
#     'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2, 0.3],
#     'max_depth': [3, 4, 5],
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

# Space
space = {
    'learning_rate': hp.choice('learning_rate', [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 3]),
    'max_depth': hp.choice('max_depth', [1, 2, 3]),
    'gamma': hp.choice('gamma', [i/10.0 for i in range(0, 5)]),
    'colsample_bytree': hp.choice('colsample_bytree', [i/10.0 for i in range(3, 10)]),
    'reg_alpha': hp.choice('reg_alpha', [1e-5, 1e-2, 0.1, 1, 10, 100]),
    'reg_lambda': hp.choice('reg_lambda', [1e-5, 1e-2, 0.1, 1, 10, 100])
}


# %%
all_params = []
all_losses = []
all_cm = []

# Objective function


def objective(params):
    # Initializations
    xgboost = XGBClassifier(objective='binary:logistic', seed=0, **params)
    scores = []
    params_log = []

    # Manually split data and compute scores
    for train_index, test_index in tss.split(data[FEATURES]):
        X_train, X_test = data[FEATURES].iloc[train_index], data[FEATURES].iloc[test_index]
        y_train, y_test = data[TARGET].iloc[train_index], data[TARGET].iloc[test_index]

        # Normalize features
        X_train = pd.DataFrame(sc.fit_transform(
            X_train), index=X_train.index, columns=X_train.columns)
        X_test = pd.DataFrame(sc.transform(
            X_test), index=X_test.index, columns=X_test.columns)

        xgboost.fit(X_train, y_train)
        predictions = xgboost.predict(X_test)
        score = accuracy_score(y_test, predictions)
        scores.append(score)
        params_log.append(params)

        # Compute and store the confusion matrix
        cm = confusion_matrix(y_test, predictions)

        # Normalize the confusion matrix to show percentages
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        all_cm.append(cm_normalized)

    # Calculate mean score
    mean_score = np.mean(scores)

    # Loss is negative score
    loss = -mean_score

    all_params.append(params)
    all_losses.append(loss)

    # Dictionary with information for evaluation
    return {'loss': loss, 'params': params_log, 'status': STATUS_OK}


# %%
# for train_idx, val_idx in tss.split(data):
# Optimize
best = fmin(fn=objective, space=space, algo=tpe.suggest,
            max_evals=48, trials=Trials())

# %%
df = pd.DataFrame(all_params)
df['loss'] = all_losses
df['normalized_loss'] = (df['loss'] - df['loss'].min()) / \
    (df['loss'].max() - df['loss'].min())


fig = px.parallel_coordinates(
    df,
    color="normalized_loss",
    labels={col: col for col in df.columns},
    color_continuous_scale=px.colors.diverging.Tealrose,
    color_continuous_midpoint=2
)

fig.show()

# %%
print(all_cm)
# for index, cm in enumerate(all_cm):
#     fig = ff.create_annotated_heatmap(cm, colorscale='Viridis')
#     fig.update_layout(title=f"Confusion Matrix for Fold {index+1}")
#     fig.show()
