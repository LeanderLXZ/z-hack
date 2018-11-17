import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor




class ModelBase(object):
    """
        Base Model Class of Models in scikit-learn Module
    """
    def __init__(self, x, y, pred_head, forecast_num):
        self.x = x
        self.y = y
        self.pred_head = pred_head
        self.forecast_num = forecast_num

    @staticmethod
    def get_reg(parameters):
        return DecisionTreeRegressor()

    def fit(self, parameters=None, early_stopping_rounds=1000):
        # Get Classifier
        reg = self.get_reg(parameters)
        # Training Model
        reg.fit(self.x, self.y)
        return reg

    @staticmethod
    def predict(reg, head, forecast_num):
        """
        :param reg: regressor
        :param head: np.array
        :param forecast_num: forecast
        :return: prediction series
        """
        pred_list = []
        for i in range(forecast_num):
            pred_i = reg.predict(head)[0]
            head[:-1], head[-1] = head[1:], pred_i
            pred_list.append(pred_i)
        return np.array(pred_list)

    @staticmethod
    def get_importance(reg):
        print('------------------------------------------------------')
        print('Feature Importance')
        importance = reg.feature_importances_
        indices = np.argsort(importance)[::-1]
        feature_num = len(importance)
        for f in range(feature_num):
            print("%d | feature %d | %d" % (
                f + 1, indices[f], importance[indices[f]]))

    def train(self, parameters=None, cv_generator=None,
              show_importance=False, early_stopping_rounds=1000):
        if cv_generator:
            pred = None
        else:
            # Fitting and Training Model
            reg = self.fit(parameters, early_stopping_rounds=early_stopping_rounds)
            # Feature Importance
            if show_importance:
                self.get_importance(reg)
            # Predicting
            pred = self.predict(reg, self.pred_head, self.forecast_num)
        return pred


class KNearestNeighbor(ModelBase):
    """
        k-Nearest Neighbor Regressor
    """
    @staticmethod
    def get_reg(parameters):
        return KNeighborsRegressor(**parameters)


class SupportVectorMachine(ModelBase):
    """
        SVM - Support Vector Machine
    """
    @staticmethod
    def get_reg(parameters):
        return SVR(**parameters)


class DecisionTree(ModelBase):
    """
        Decision Tree
    """
    @staticmethod
    def get_reg(parameters):
        return DecisionTreeRegressor(**parameters)

class RandomForest(ModelBase):
    """
        Random forecast
    """
    @staticmethod
    def get_reg(parameters):
        return RandomForestRegressor(**parameters)


class ExtraTrees(ModelBase):
    """
        Extra Trees
    """
    @staticmethod
    def get_reg(parameters):
        return ExtraTreesRegressor(**parameters)


class AdaBoost(ModelBase):
    """
        AdaBoost
    """
    @staticmethod
    def get_reg(parameters):
        return AdaBoostRegressor(**parameters)


class GradientBoosting(ModelBase):
    """
        Gradient Boosting
    """
    @staticmethod
    def get_reg(parameters):
        return GradientBoostingRegressor(**parameters)

class XGBoost(ModelBase):
    """
        XGBoost using sklearn module
    """
    @staticmethod
    def get_reg(parameters=None):
        return XGBRegressor(**parameters)

    def fit(self, parameters=None, early_stopping_rounds=1000):
        reg = self.get_reg(parameters)
        reg.fit(self.x, self.y,
                eval_set=[(self.x, self.y)],
                early_stopping_rounds=early_stopping_rounds,
                eval_metric='mae', verbose=True)
        return reg


class LightGBM(ModelBase):
    """
        LightGBM using sklearn module
    """
    @staticmethod
    def get_reg(parameters=None):
        return LGBMRegressor(**parameters)

    def fit(self, parameters=None, early_stopping_rounds=1000):
        reg = self.get_reg(parameters)
        reg.fit(self.x, self.y,
                eval_set=[(self.x, self.y)], eval_names=['train'],
                early_stopping_rounds=early_stopping_rounds,
                eval_metric='mae', verbose=True)
        return reg


class MachineLearningModel(object):

    def __init__(self, x, y, pred_head, forecast_num):
        self.x = x
        self.y = y
        self.pred_head = pred_head
        self.forecast_num = forecast_num

    def predict(self, model_name):

        train_seed = 95

        if model_name == 'knn':
            parameters ={'n_neighbors': 5,
                         'weights': 'uniform',
                         'algorithm': 'auto',
                         'leaf_size': 30,
                         'p': 2,
                         'metric': 'minkowski',
                         'metric_params': None,
                         'n_jobs': -1}
            pred = KNearestNeighbor(
                self.x, self.y, self.pred_head, self.forecast_num).train(
                parameters)
        elif model_name == 'svm':
            parameters = {'kernel': 'rbf',
                          'degree': 3,
                          'gamma': 'auto',
                          'coef0': 0.0,
                          'tol': 0.001,
                          'C': 1.0,
                          'epsilon': 0.1,
                          'shrinking': True,
                          'cache_size': 200,
                          'verbose': False,
                          'max_iter': -1}
            pred = SupportVectorMachine(
                self.x, self.y, self.pred_head, self.forecast_num).train(
                parameters)
        elif model_name == 'dt':
            parameters = {'criterion': 'mae',
                          'splitter': 'best',
                          'max_depth': None,
                          'min_samples_split': 2,
                          'min_samples_leaf': 1,
                          'min_weight_fraction_leaf': 0.0,
                          'max_features': None,
                          'random_state': train_seed,
                          'max_leaf_nodes': None,
                          'min_impurity_decrease': 0.0,
                          'min_impurity_split': None,
                          'presort': False}
            pred = DecisionTree(
                self.x, self.y, self.pred_head, self.forecast_num).train(
                parameters)
        elif model_name == 'rf':
            parameters = {'bootstrap': True,
                          'criterion': 'mae',
                          'max_depth': None,
                          'max_features': 'auto',
                          'max_leaf_nodes': None,
                          'min_impurity_decrease': 0.0,
                          'min_samples_leaf': 1,
                          'min_samples_split': 2,
                          'min_weight_fraction_leaf': 0.0,
                          'n_estimators': 10,
                          'n_jobs': -1,
                          'oob_score': True,
                          'random_state': train_seed,
                          'verbose': 0,
                          'warm_start': False}
            pred = RandomForest(
                self.x, self.y, self.pred_head, self.forecast_num).train(
                parameters)
        elif model_name == 'et':
            parameters = {'n_estimators': 10,
                          'criterion': 'mae',
                          'max_depth': None,
                          'min_samples_split': 2,
                          'min_samples_leaf': 1,
                          'min_weight_fraction_leaf': 0.0,
                          'max_features': 'auto',
                          'max_leaf_nodes': None,
                          'min_impurity_decrease': 0.0,
                          'min_impurity_split': None,
                          'bootstrap': False,
                          'oob_score': False,
                          'n_jobs': -1,
                          'random_state': train_seed,
                          'verbose': 0,
                          'warm_start': False}
            pred = ExtraTrees(
                self.x, self.y, self.pred_head, self.forecast_num).train(
                parameters)
        elif model_name == 'ab':
            et_parameters = {'n_estimators': 10,
                             'criterion': 'mae',
                             'max_depth': None,
                             'min_samples_split': 2,
                             'min_samples_leaf': 1,
                             'min_weight_fraction_leaf': 0.0,
                             'max_features': 'auto',
                             'max_leaf_nodes': None,
                             'min_impurity_decrease': 0.0,
                             'min_impurity_split': None,
                             'bootstrap': False,
                             'oob_score': False,
                             'n_jobs': -1,
                             'random_state': train_seed,
                             'verbose': 0,
                             'warm_start': False}
            reg_et = ExtraTreesRegressor(**et_parameters)
            parameters = {'loss': 'linear',
                          'base_estimator': reg_et,
                          'learning_rate': 1.0,
                          'n_estimators': 50,
                          'random_state': train_seed}
            pred = AdaBoost(
                self.x, self.y, self.pred_head, self.forecast_num).train(
                parameters)
        elif model_name == 'gb':
            parameters = {'loss': 'ls',
                          'learning_rate': 0.1,
                          'n_estimators': 100,
                          'subsample': 1.0,
                          'criterion': 'mae',
                          'min_samples_split': 2,
                          'min_samples_leaf': 1,
                          'min_weight_fraction_leaf': 0.0,
                          'max_depth': 3,
                          'min_impurity_decrease': 0.0,
                          'min_impurity_split': None,
                          'init': None,
                          'random_state': train_seed,
                          'max_features': None,
                          'alpha': 0.9,
                          'verbose': 0,
                          'max_leaf_nodes': None,
                          'warm_start': False,
                          'presort': 'auto'}
            pred = GradientBoosting(
                self.x, self.y, self.pred_head, self.forecast_num).train(
                parameters)
        elif model_name == 'xgb':
            parameters = {'max_depth': 3,
                          'learning_rate': 0.1,
                          'n_estimators': 100,
                          'silent': True,
                          'objective': 'reg:linear',
                          'booster': 'gbtree',
                          'n_jobs': -1,
                          'nthread': None,
                          'gamma': 0,
                          'min_child_weight': 1,
                          'max_delta_step': 0,
                          'subsample': 1,
                          'colsample_bytree': 1,
                          'colsample_bylevel': 1,
                          'reg_alpha': 0,
                          'reg_lambda': 1,
                          'scale_pos_weight': 1,
                          'base_score': 0.5,
                          'random_state': train_seed,
                          'seed': train_seed,
                          'missing': None,
                          'importance_type': 'gain'}
            pred = XGBoost(
                self.x, self.y, self.pred_head, self.forecast_num).train(
                parameters, early_stopping_rounds=None)
        elif model_name == 'lgb':
            parameters = {'boosting_type': 'gbdt',
                          'num_leaves': 31,
                          'max_depth': -1,
                          'learning_rate': 0.1,
                          'n_estimators': 100,
                          'subsample_for_bin': 200000,
                          'objective': 'regression',
                          'class_weight': None,
                          'min_split_gain': 0.0,
                          'min_child_weight': 0.001,
                          'min_child_samples': 20,
                          'subsample': 1.0,
                          'subsample_freq': 0,
                          'colsample_bytree': 1.0,
                          'reg_alpha': 0.0,
                          'reg_lambda': 0.0,
                          'random_state': train_seed,
                          'n_jobs': -1,
                          'silent': True,
                          'importance_type': 'split'}
            pred = LightGBM(
                self.x, self.y, self.pred_head, self.forecast_num).train(
                parameters, early_stopping_rounds=None)
        else:
            raise Exception("Wrong Model Name!")

        return pred
