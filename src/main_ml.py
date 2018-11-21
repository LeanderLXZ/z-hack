import time
from models import utils
import pandas as pd
import numpy as np
from os.path import join
from models.regressors import MachineLearningModel
from config import cfg


class Training(object):

    def __init__(self,
                 sample_mode,
                 select_col,
                 fill_mode=None):

        self.sample_mode = sample_mode
        self.select_col = select_col
        self.suffix = utils.get_suffix(sample_mode)
        self.fill_mode = fill_mode

        file_path = cfg.preprocessed_path
        if fill_mode:
            print('Loading {}...'.format(
                join(file_path, 'total_day_{}.csv'.format(fill_mode))))
            self.data = pd.read_csv(
                join(file_path, 'total_day_{}.csv'.format(fill_mode)),
                index_col='DATE',
                parse_dates=['DATE'])
        else:
            print('Loading {}...'.format(
                join(file_path, 'total_{}.csv'.format(sample_mode))))
            self.data = pd.read_csv(
                join(file_path, 'total_{}.csv'.format(sample_mode)),
                index_col='DATE',
                parse_dates=['DATE'])
            # print(self.data)

        self.month_features = utils.get_month_features()

    @staticmethod
    def _train_valid_split(features,
                           labels,
                           final_head,
                           date_list,
                           train_start,
                           valid_start,
                           valid_end):
        if train_start not in date_list:
            train_start = date_list[0]
        train_started = False
        valid_started = False
        valid_ended = False
        train_features = []
        train_labels = []
        valid_head = None
        valid_labels = []
        final_features = []
        final_labels = []
        for idx, date in enumerate(date_list):
            if date == train_start:
                train_started = True
            if train_started:
                final_features.append(features[idx])
                final_labels.append(labels[idx])
            if date == valid_start:
                valid_head = features[idx]
                valid_started = True
            if train_started and not valid_started:
                train_features.append(features[idx])
                train_labels.append(labels[idx])
            if valid_started and not valid_ended:
                valid_labels.append(labels[idx])
            if date == valid_end:
                valid_ended = True
            if idx == len(date_list) - 1:
                assert labels[idx] == final_head[-1]
        return np.array(train_features), np.array(train_labels), \
               np.array(valid_head), np.array(valid_labels),\
               np.array(final_features), np.array(final_labels)

    def series_to_features(self,
                           feature_num,
                           month_features=None,
                           time_features=None):
        series = self.data[
            '{}_{}'.format(self.select_col, self.suffix)].values
        date = list(map(lambda x: x.strftime('%Y-%m-%d'),
                        self.data.index.to_pydatetime()))
        features = []
        labels = []
        date_list = []
        pred_head = None
        feature_idx_start = 0
        feature_idx_end = np.max((feature_num, np.max(time_features)))
        while feature_idx_end < len(series):
            feature = []
            date_list.append(date[feature_idx_end])
            for f_i, feature_num_i in enumerate(time_features):
                feature.append(np.mean(
                    series[feature_idx_end-feature_num_i:feature_idx_end]))
            feature = np.append(
                feature, series[feature_idx_start:feature_idx_end])
            if month_features:
                month = self.data.index[feature_idx_end]
                month_str = month.to_pydatetime().strftime('%Y-%m')
                feature = np.append(
                    month_features['pre'][month_str], feature)
            features.append(feature)
            labels.append(series[feature_idx_end])
            if feature_idx_end == len(series) - 1:
                feature = []
                for f_i, feature_num_i in enumerate(time_features):
                    feature.append(np.mean(labels[-feature_num_i:]))
                pred_head = np.append(
                    feature, series[feature_idx_start+1:feature_idx_end+1])
                if month_features:
                    month = self.data.index[feature_idx_end]
                    month_str = month.to_pydatetime().strftime('%Y-%m')
                    pred_head = np.append(
                        month_features['now'][month_str], pred_head)
            feature_idx_start += 1
            feature_idx_end += 1
        return np.array(features), np.array(labels), \
               np.array(pred_head), date_list

    @staticmethod
    def _calc_acc(y, pred):
        if pred.sum() < 1:
            return np.nan
        assert len(y) == len(pred), (len(y), len(pred))
        cost = []
        for y_i, pred_i in zip(y, pred):
            cost.append(abs(y_i - pred_i))
        return np.mean(cost)

    def save_result(self,
                    pred,
                    model_name,
                    append_info=None):

        utils.check_dir([cfg.result_path])
        df_pred = pd.read_csv(
            join(cfg.source_path, 'z_hack_submit_new.csv'),
            index_col=['FORECASTDATE'], usecols=['FORECASTDATE'])
        assert len(df_pred) == len(pred), (len(df_pred), len(pred))
        df_pred['FORECASTPRICE'] = pred
        if append_info is None:
            append_info = '_' + model_name + '_' + self.sample_mode
        df_pred.to_csv(
            join(cfg.result_path, 'result{}.csv'.format(append_info)),
            float_format='%.2f')

    def get_shifted_results(self,
                            model_name,
                            x_final,
                            y_final,
                            head_final,
                            time_features=None,
                            parameters=None,
                            use_month_features=False,
                            append_info=None):

        # Get predictions for different lengths for different fill modes
        if self.fill_mode in ['w_ff', 'w_bf', 'w_avg', 'w_line']:
            pred_final_sf = MachineLearningModel(
                x_final, y_final, head_final,
                forecast_num=22
            ).predict(
                model_name,
                time_features=time_features,
                use_month_features=use_month_features,
                model_parameters=parameters
            )
            df_pred_sf = pd.read_csv(
                join(cfg.source_path, 'z_hack_submit_new_day_work.csv'),
                index_col=['FORECASTDATE'], usecols=['FORECASTDATE']
            )
        elif self.fill_mode in ['a_ff', 'a_bf', 'a_avg', 'a_line']:
            pred_final_sf = MachineLearningModel(
                x_final, y_final, head_final,
                forecast_num=30
            ).predict(
                model_name,
                time_features=time_features,
                use_month_features=use_month_features,
                model_parameters=parameters
            )
            df_pred_sf = pd.read_csv(
                join(cfg.source_path, 'z_hack_submit_new_day_all.csv'),
                index_col=['FORECASTDATE'], usecols=['FORECASTDATE']
            )
        else:
            raise ValueError('Wrong Fill Mode! Find: {}'.format(self.fill_mode))

        # Assign prediction to DataFrame
        assert len(df_pred_sf) == len(pred_final_sf), \
            (len(df_pred_sf), len(pred_final_sf))
        df_pred_sf['FORECASTPRICE'] = pred_final_sf

        # Reindex DataFrame to get final results
        df_pred = pd.read_csv(
            join(cfg.source_path, 'z_hack_submit_new.csv'),
            index_col=['FORECASTDATE'], usecols=['FORECASTDATE'])
        df_pred_sf = df_pred_sf.reindex(df_pred.index)

        # Save shifted results
        assert len(df_pred_sf) == len(df_pred), \
            (len(df_pred_sf), len(df_pred))
        utils.check_dir([cfg.result_path])
        shifted_result_path = join(cfg.result_path, 'shifted_results')
        utils.check_dir([shifted_result_path])
        df_pred_sf.to_csv(join(
            shifted_result_path, 'result_sf{}.csv'.format(append_info)),
            float_format='%.2f')


    def train(self,
              model_name,
              parameters=None,
              feature_num=50,
              forecast_num=21,
              time_features=None,
              use_month_features=False,
              data_range=None,
              save_result=False,
              save_shifted_result=False,
              append_info=None):

        if use_month_features:
            month_features = self.month_features
        else:
            month_features = None

        features, labels, final_head, date_list = \
            self.series_to_features(
                feature_num,
                month_features=month_features,
                time_features=time_features
            )
        train_features, train_labels, valid_head, \
            valid_labels, final_features, final_labels = \
            self._train_valid_split(
                features, labels, final_head, date_list,
                train_start=data_range['train_start'],
                valid_start=data_range['valid_start'],
                valid_end=data_range['valid_end']
            )
        # print(train_features[0])
        # print(train_features, train_labels, valid_head,
        #       valid_labels, final_features, final_labels)
        pred_valid = MachineLearningModel(
            train_features, train_labels, valid_head,
            forecast_num=len(valid_labels)
        ).predict(
            model_name,
            time_features=time_features,
            use_month_features=use_month_features,
            model_parameters=parameters
        )
        pred_final = MachineLearningModel(
            final_features, final_labels, final_head,
            forecast_num=forecast_num
        ).predict(
            model_name,
            time_features=time_features,
            use_month_features=use_month_features,
            model_parameters=parameters
        )
        cost = self._calc_acc(valid_labels, pred_valid)

        # Save shifted results for different fill modes
        if (self.fill_mode is not None) and save_shifted_result:
            self.get_shifted_results(
                model_name, final_features, final_labels, final_head,
                parameters=parameters,
                time_features=time_features,
                use_month_features=use_month_features,
                append_info=append_info)

        # Save final results
        if save_result:
            self.save_result(pred_final, model_name, append_info)

        return pred_final, cost, pred_valid


if __name__ == '__main__':

    start_time = time.time()
    utils.check_dir([cfg.log_path])
    df = pd.read_csv(join(cfg.source_path, 'z_hack_submit_new.csv'),
                     index_col=['FORECASTDATE'], usecols=['FORECASTDATE'])
    T = Training('day', 'CONTPRICE')
    range_1 = {'train_start': '2009-01-04',
               'valid_start': '2013-12-02',
               'valid_end': '2013-12-31'}

    for model_i in ('knn', 'svm', 'dt', 'rf',
                    'et', 'ab', 'gb', 'xgb', 'lgb'):
        model_start_time = time.time()
        utils.thin_line()
        print('Start Model: {}'.format(model_i))
        df['{}_day'.format(model_i)], _, _ = T.train(
            model_i,
            feature_num=21,
            forecast_num=21,
            time_features=(10, 20, 30, 40, 50),
            use_month_features=True,
            data_range=range_1,
            save_result=True
        )
        print('Model {} Done! Using time {:.2f}s...'.format(
            model_i, time.time() - model_start_time))

    df.to_csv(join(cfg.log_path, 'result_day.csv'))
    utils.thick_line()
    print('All Done! Using time {:.2f}s...'.format(
        time.time() - start_time))
    utils.thick_line()
