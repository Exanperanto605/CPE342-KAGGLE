import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

id_list = test_df["id"]

boolean_features = ["guild_membership", "is_premium_member", "owns_limited_edition"]


def calc_normalized_mae(y_true, y_pred) -> float:
    mean_y_true = np.mean(y_true)
    if mean_y_true == 0:
        return 0

    mae = mean_absolute_error(y_true, y_pred)
    normalized_mae = mae / mean_y_true

    return normalized_mae


# def task3():
if __name__ == "__main__":  # PLACEHOLDER
    # Step 1: Data Preprocessing
    train_df.drop(["id", "player_id"], axis=1, inplace=True)
    test_df.drop(["id", "player_id"], axis=1, inplace=True)

    # - Impute missing values in boolean features
    train_df[boolean_features] = train_df[boolean_features].fillna(0)
    test_df[boolean_features] = test_df[boolean_features].fillna(0)

    # - Impute missing values in the remaining features with the median value of the feature
    median_float_imputer = SimpleImputer(strategy="median")

    train_df = pd.DataFrame(median_float_imputer.fit_transform(train_df), columns=train_df.columns)
    test_df = pd.DataFrame(median_float_imputer.fit_transform(test_df), columns=test_df.columns)

    # - Separate the target variable from the dependent variables
    X = train_df.drop("spending_30d", axis=1)
    y = train_df["spending_30d"]

    # Step 2: Model Training
    # - Define constant variables
    TEST_SIZE = 0.25        # default = 0.33    previous = 0.25
    RANDOM_STATE = 42       # default = 42      previous = 42

    MAX_DEPTH = 8           # default = 3       previous = 8
    LEARNING_RATE = 0.15    # default = 0.1     previous = 0.20

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    # model = LinearRegression()
    model = GradientBoostingRegressor(loss="absolute_error", max_depth=MAX_DEPTH, learning_rate=LEARNING_RATE)

    model.fit(X_train, y_train)

    # Step 3: Model Testing & Evaluation
    y_pred = model.predict(X_test)

    n_mae = calc_normalized_mae(y_test, y_pred)
    print(f"Normalized Mean Absolute Error: {n_mae:.5f}")   # Current Best: 0.27946

    result = model.predict(test_df)

    result_df = pd.DataFrame(data={"id": id_list, "task3": result})

    print(result_df)

    result_df.to_csv("task3_submission.csv", index=False)

    # return result_df
