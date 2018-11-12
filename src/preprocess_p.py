import time
import numpy as np
import pandas as pd
from models import utils
from math import ceil
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelBinarizer
from config import cfg

group_list = None


class DataPreProcess(object):

    def __init__(self, train_path, test_path, preprocess_path, use_group_list=None, use_code_id=False,
                 add_train_dummies=False, merge_eras=False, use_global_valid=False, global_valid_rate=None,
                 drop_outliers_by_value=False, drop_outliers_by_quantile=False, standard_scale=False,
                 min_max_scale=False, add_polynomial_features=False, generate_valid_for_fw=False,
                 split_data_by_gan=False, split_data_by_era=False):

        self.train_path = train_path
        self.test_path = test_path
        self.preprocess_path = preprocess_path
        self.x_train = pd.DataFrame()
        self.x_g_train = pd.DataFrame()
        self.y_train = pd.DataFrame()
        self.w_train = pd.DataFrame()
        self.code_id_train = pd.DataFrame()
        self.e_train = pd.DataFrame()
        self.x_test = pd.DataFrame()
        self.y_test = pd.DataFrame()
        self.w_test = pd.DataFrame()
        self.e_test = pd.DataFrame()
        self.pct_test = pd.DataFrame()
        self.x_g_test = pd.DataFrame()
        self.code_id_test = pd.DataFrame()
        self.id_test = pd.DataFrame()

        # Positive Data Set
        self.x_train_p = pd.DataFrame()
        self.x_g_train_p = pd.DataFrame()
        self.y_train_p = pd.DataFrame()
        self.w_train_p = pd.DataFrame()
        self.e_train_p = pd.DataFrame()
        self.id_test_p = pd.DataFrame()

        # Negative Data Set
        self.x_train_n = pd.DataFrame()
        self.x_g_train_n = pd.DataFrame()
        self.y_train_n = pd.DataFrame()
        self.w_train_n = pd.DataFrame()
        self.e_train_n = pd.DataFrame()
        self.id_test_n = pd.DataFrame()

        # Validation Set
        self.x_valid = np.array([])
        self.x_g_valid = np.array([])
        self.y_valid = np.array([])
        self.w_valid = np.array([])
        self.e_valid = np.array([])
        self.code_id_valid = np.array([])

        self.use_group_list = use_group_list
        self.use_code_id = use_code_id
        self.add_train_dummies_ = add_train_dummies
        self.drop_feature_list = []
        self.merge_era_range_list = merge_era_range_list
        self.merge_eras_ = merge_eras
        self.use_global_valid_ = use_global_valid
        self.global_valid_rate = global_valid_rate
        self.drop_outliers_by_value_ = drop_outliers_by_value
        self.drop_outliers_by_quantile_ = drop_outliers_by_quantile
        self.standard_scale_ = standard_scale
        self.min_max_scale_ = min_max_scale
        self.add_polynomial_features_ = add_polynomial_features
        self.generate_valid_for_fw_ = generate_valid_for_fw
        self.split_data_by_gan_ = split_data_by_gan
        self.split_data_by_era_ = split_data_by_era

        if group_list is None:
            if use_group_list is not None:
                raise ValueError("Groups Not Found!")

        if use_group_list is not None:
            self.g_train = pd.DataFrame()
            self.g_test = pd.DataFrame()
            self.g_train_dict = {}
            self.g_test_dict = {}
            if len(use_group_list) > 1:
                print('======================================================')
                print('[W] Using Multi Groups: {}'.format(use_group_list))
            else:
                print('======================================================')
                print('[W] Using Single Group: {}'.format(use_group_list[0]))

    # Load CSV Files Using Pandas
    def load_csv(self):

        train_f = pd.read_csv(self.train_path, header=0, dtype=np.float64)
        test_f = pd.read_csv(self.test_path, header=0, dtype=np.float64)

        return train_f, test_f

    # Load Data Using Pandas
    def load_data(self):

        try:
            print('======================================================')
            print('Loading data...')
            train_f, test_f = self.load_csv()
        except Exception as e:
            print('Unable to read data: ', e)
            raise

        if group_list is not None:
            self.drop_feature_list.extend(['group' + str(g) for g in group_list])
        if self.use_code_id:
            self.drop_feature_list.append('code_id')

        # Drop Unnecessary Columns
        self.x_train = train_f.drop(['index', 'weight', 'label', 'era', 'date', 'pct', *self.drop_feature_list], axis=1)
        self.y_train = train_f['label']
        self.w_train = train_f['weight']
        self.e_train = train_f['era']
        self.x_test = test_f.drop(['index', 'weight', 'label', 'era', 'date', 'pct', *self.drop_feature_list], axis=1)
        self.y_test = test_f['label']
        self.w_test = test_f['weight']
        self.e_test = test_f['era']
        self.id_test = test_f['index']
        if self.use_code_id:
            self.code_id_train = train_f['code_id']
            self.code_id_test = test_f['code_id']

        print('------------------------------------------------------')
        print('Train Features: {}\n'.format(self.x_train.shape[1]),
              'Test Features: {}'.format(self.x_test.shape[1]))

        if self.use_group_list is not None:
            for i in self.use_group_list:
                self.g_train_dict[i] = train_f['group' + str(i)]
                self.g_test_dict[i] = test_f['group' + str(i)]
                self.x_g_train = self.x_train
                self.x_g_test = self.x_test

    # Convert pandas DataFrames to numpy arrays
    def convert_pd_to_np(self):

        print('======================================================')
        print('Converting pandas DataFrames to numpy arrays...')

        self.x_train = np.array(self.x_train, dtype=np.float64)
        self.y_train = np.array(self.y_train, dtype=np.float64)
        self.w_train = np.array(self.w_train, dtype=np.float64)
        self.e_train = np.array(self.e_train, dtype=int)
        self.x_test = np.array(self.x_test, dtype=np.float64)
        self.y_test = np.array(self.y_test, dtype=np.float64)
        self.w_test = np.array(self.w_test, dtype=np.float64)
        self.e_test = np.array(self.e_test, dtype=int)
        self.id_test = np.array(self.id_test, dtype=int)

    # Add Polynomial Features
    def add_polynomial_features(self):

        print('======================================================')
        print('Adding Polynomial Features...')

        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)

        self.x_train = poly.fit_transform(self.x_train)
        self.x_test = poly.fit_transform(self.x_test)

    # Convert Column 'group' to Dummies
    def convert_group_to_dummies(self, add_train_dummies=False):

        print('======================================================')
        print('Converting Groups of Train Set to Dummies...')

        lb_dict = {}
        for i in self.use_group_list:
            lb = LabelBinarizer()
            if add_train_dummies:
                print('------------------------------------------------------')
                print('Add Zero Dummies to Train Set of Group: {}'.format(i))
                add_list = list((set(self.g_train_dict[i]) ^ set(self.g_test_dict[i])) & set(self.g_test_dict[i]))
                lb.fit(np.append(self.g_train_dict[i], add_list))
            else:
                lb.fit(self.g_train_dict[i])
            lb_dict[i] = lb

        for i in self.use_group_list:

            print('------------------------------------------------------')
            print('Converting Group {} to Dummies...'.format(i))

            train_dummies = lb_dict[i].transform(self.g_train_dict[i])
            test_dummies = lb_dict[i].transform(self.g_test_dict[i])

            print('Train Dummies: {}\n'.format(train_dummies.shape[1]),
                  'Test Dummies: {}'.format(test_dummies.shape[1]))

            if self.x_train.shape[1] > 500:
                print('So Many Features!')
                for ii in range(len(self.x_train)):
                    self.x_train[ii] = np.concatenate((self.x_train[ii], train_dummies[ii]))
                for ii in range(len(self.x_test)):
                    self.x_test[i] = np.concatenate((self.x_test[ii], test_dummies[ii]))
            else:
                self.x_train = np.concatenate((self.x_train, train_dummies), axis=1)
                self.x_test = np.concatenate((self.x_test, test_dummies), axis=1)

            self.x_g_train = np.column_stack((self.x_g_train, self.g_train_dict[i]))
            self.x_g_test = np.column_stack((self.x_g_test, self.g_test_dict[i]))

        print('------------------------------------------------------')
        print('Total Features of x_train: {}\n'.format(self.x_train.shape[1]),
              'Total Features of x_test: {}\n'.format(self.x_test.shape[1]),
              'Total Features of x_g_train: {}\n'.format(self.x_g_train.shape[1]),
              'Total Features of x_g_test: {}'.format(self.x_g_test.shape[1]))

    # Spilt Validation Set by valid_rate
    def split_validation_set(self, valid_rate=None):

        print('======================================================')
        print('Splitting Validation Set by Valid Rate: {}'.format(valid_rate))

        n_era = len(set(self.e_train))
        n_era_valid = ceil(valid_rate*n_era)
        valid_era = list(range(n_era))[-n_era_valid:]

        print('Number of Eras: {}\n'.format(n_era),
              'Number of Valid Eras: {}\n'.format(n_era_valid),
              'Valid Eras: {}-{}'.format(valid_era[0], valid_era[-1]))

        train_index = []
        valid_index = []

        # Generate train-validation split index
        for ii, ele in enumerate(self.e_train):
            if ele in valid_era:
                valid_index.append(ii)
            else:
                train_index.append(ii)

        # Validation Set
        self.x_valid = self.x_train[valid_index]
        self.y_valid = self.y_train[valid_index]
        self.w_valid = self.w_train[valid_index]
        self.e_valid = self.e_train[valid_index]

        # Train Set
        self.x_train = self.x_train[train_index]
        self.y_train = self.y_train[train_index]
        self.w_train = self.w_train[train_index]
        self.e_train = self.e_train[train_index]

        if group_list is not None:
            self.x_g_valid = self.x_g_train[valid_index]
            self.x_g_train = self.x_g_train[train_index]
        if self.use_code_id:
            self.code_id_valid = self.code_id_train[valid_index]
            self.code_id_train = self.code_id_train[train_index]

    # Save Data
    def save_data(self):

        print('======================================================')
        print('Saving Preprocessed Data...')
        utils.save_data_to_pkl(self.x_train, self.preprocess_path + 'x_train.p')
        utils.save_data_to_pkl(self.y_train, self.preprocess_path + 'y_train.p')
        utils.save_data_to_pkl(self.w_train, self.preprocess_path + 'w_train.p')
        utils.save_data_to_pkl(self.e_train, self.preprocess_path + 'e_train.p')
        utils.save_data_to_pkl(self.y_test, self.preprocess_path + 'y_test.p')
        utils.save_data_to_pkl(self.x_test, self.preprocess_path + 'x_test.p')
        utils.save_data_to_pkl(self.w_test, self.preprocess_path + 'w_test.p')
        utils.save_data_to_pkl(self.e_test, self.preprocess_path + 'e_test.p')
        utils.save_data_to_pkl(self.id_test, self.preprocess_path + 'id_test.p')

        if group_list is not None:
            utils.save_data_to_pkl(self.x_g_train, self.preprocess_path + 'x_g_train.p')
            utils.save_data_to_pkl(self.x_g_test, self.preprocess_path + 'x_g_test.p')
        if self.use_code_id:
            utils.save_data_to_pkl(self.code_id_train, self.preprocess_path + 'code_id_train.p')
            utils.save_data_to_pkl(self.code_id_test, self.preprocess_path + 'code_id_test.p')

    # Save Validation Set
    def save_global_valid_set(self):

        print('======================================================')
        print('Saving Validation Set...')
        utils.save_data_to_pkl(self.x_valid, self.preprocess_path + 'x_global_valid.p')
        utils.save_data_to_pkl(self.y_valid, self.preprocess_path + 'y_global_valid.p')
        utils.save_data_to_pkl(self.w_valid, self.preprocess_path + 'w_global_valid.p')
        utils.save_data_to_pkl(self.e_valid, self.preprocess_path + 'e_global_valid.p')

        if group_list is not None:
            utils.save_data_to_pkl(self.x_g_valid, self.preprocess_path + 'x_g_global_valid.p')
        if self.use_code_id:
            utils.save_data_to_pkl(self.code_id_valid, self.preprocess_path + 'code_id_global_valid.p')

    # Preprocess
    def preprocess(self):

        print('======================================================')
        print('Start Preprocessing...')

        start_time = time.time()

        # Load original data
        self.load_data()

        # Convert pandas DataFrames to numpy arrays
        self.convert_pd_to_np()

        # # Add Polynomial Features
        # if self.add_polynomial_features_:
        #     self.add_polynomial_features()

        # Convert column 'group' to dummies
        if group_list is not None:
            self.convert_group_to_dummies(add_train_dummies=self.add_train_dummies_)

        # Spilt Validation Set by valid_rate
        if self.use_global_valid_:
            self.split_validation_set(valid_rate=self.global_valid_rate)
            self.save_global_valid_set()

        # Save Data to pickle files
        self.save_data()

        end_time = time.time()

        print('======================================================')
        print('Done!')
        print('Using {:.3}s'.format(end_time - start_time))
        print('======================================================')


if __name__ == '__main__':

    utils.check_dir(['./data/', cfg.preprocessed_path])

    preprocess_args = {'merge_eras': False,
                       'use_code_id': False,
                       'use_group_list': None,  # Should be a list or None
                       'add_train_dummies': False,
                       'use_global_valid': False,
                       'generate_valid_for_fw': False,
                       'global_valid_rate': 0.1,
                       'drop_outliers_by_value': False,
                       'drop_outliers_by_quantile': False,
                       'standard_scale': False,
                       'min_max_scale': False,
                       'add_polynomial_features': False,
                       'split_data_by_gan': False,
                       'split_data_by_era': False}

    DPP = DataPreProcess(cfg.train_csv_path, cfg.test_csv_path, cfg.preprocessed_path, **preprocess_args)
    DPP.preprocess()
