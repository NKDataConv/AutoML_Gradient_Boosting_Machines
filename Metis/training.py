import numpy as np
from sklearn.datasets import fetch_openml
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import nni


def load_data():
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

    # train, test, val split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.375, random_state=1)
    X_val_1, X_val_2, y_val_1, y_val_2 = train_test_split(X_val, y_val, test_size=0.5, random_state=1)

    # prepare for lgb
    train = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    test = lgb.Dataset(X_test, label=y_test, free_raw_data=False)
    val_1 = lgb.Dataset(X_val_1, label=y_val_1, free_raw_data=False)
    val_2 = lgb.Dataset(X_val_2, label=y_val_2, free_raw_data=False)

    return train, test, val_1, val_2


def my_accuracy(y_pred, data):
    '''Calculate accuracy for evaluation in lgb training.
    Reshaping according to https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.train.html'''

    y_pred = np.reshape(y_pred, (10, -1)).T
    y_pred = np.argmax(y_pred, axis=1)
    acc = accuracy_score(data.get_label(), y_pred)

    return ("acc", acc, True)


def generate_params():
    '''Generate default parameter for lightgbm'''

    return {
        "num_boost_round": 100,
        "early_stopping_rounds": 3,
        "max_depth": 20,
        "learning_rate": 0.1324190937645,
        "num_leaves": 32,
        "min_data_in_leaf": 8,
        "max_bin": 331,
        "subsample": 0.7267958740030672,
        "min_sum_hessian_in_leaf": 0.07084497510468116,
        "feature_fraction": 0.30057074705023934,
        "lambda_l1": 20,
        "lambda_l2": 20,
        "num_class": 10,
        "objective": "multiclass"
    }


if __name__ == '__main__':

    RECEIVED_PARAMS = nni.get_next_parameter()
    params = generate_params()
    params.update(RECEIVED_PARAMS)

    train, test, val_1, val_2 = load_data()

    model = lgb.train(params, feval=my_accuracy, train_set=train, valid_sets=val_1,
                      early_stopping_rounds=params["early_stopping_rounds"],
                      num_boost_round=params["num_boost_round"])

    # obtain validation score on val_2
    preds = model.predict(val_2.data)
    final_score = accuracy_score(val_2.label.astype("int"), np.argmax(preds, axis=1))

    nni.report_final_result(final_score)
