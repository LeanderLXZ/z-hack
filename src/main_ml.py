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

    def _train_valid_split(self,
                           train_start,
                           valid_start,
                           valid_end):

        if self.sample_mode == 'day':
            train_started = False
            valid_started = False
            valid_ended = False
            train_data = []
            valid_data = []
            final_data = []
            for row_iter in self.data.iterrows():
                row = row_iter[1]
                date_str = row_iter[0].to_pydatetime().strftime('%Y-%m-%d')
                if date_str == train_start:
                    train_started = True
                if train_started:
                    final_data.append(
                        row['{}_{}'.format(self.select_col, self.suffix)])
                if date_str == valid_start:
                    valid_started = True
                if train_started and not valid_started:
                    train_data.append(
                        row['{}_{}'.format(self.select_col, self.suffix)])
                if valid_started and not valid_ended:
                    valid_data.append(
                        row['{}_{}'.format(self.select_col, self.suffix)])
                if date_str == valid_end:
                    valid_ended = True
            return np.array(train_data), np.array(valid_data), \
                   np.array(final_data)

    @staticmethod
    def series_to_features(series, feature_num):
        features = []
        label = []
        pred_head = None
        idx_start = 0
        idx_end = feature_num
        while (idx_end + 1) < len(series):
            features.append(series[idx_start:idx_end])
            label.append(series[idx_end + 1])
            idx_start += 1
            idx_end += 1
            if idx_end == len(series) - 1:
                pred_head = [series[idx_start:idx_end]]
        assert len(features[0]) == feature_num
        assert len(pred_head[0]) == feature_num
        return np.array(features), np.array(label), np.array(pred_head)

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
                            append_info=None):

        # Get predictions for different lengths for different fill modes
        if self.fill_mode in ['w_ff', 'w_bf', 'w_avg', 'w_line']:
            pred_final_sf = MachineLearningModel(
                x_final, y_final, head_final,
                forecast_num=22).predict(model_name)
            df_pred_sf = pd.read_csv(
                join(cfg.source_path, 'z_hack_submit_new_day_work.csv'),
                index_col=['FORECASTDATE'], usecols=['FORECASTDATE'])
        elif self.fill_mode in ['a_ff', 'a_bf', 'a_avg', 'a_line']:
            pred_final_sf = MachineLearningModel(
                x_final, y_final, head_final,
                forecast_num=30).predict(model_name)
            df_pred_sf = pd.read_csv(
                join(cfg.source_path, 'z_hack_submit_new_day_all.csv'),
                index_col=['FORECASTDATE'], usecols=['FORECASTDATE'])
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
              feature_num=50,
              forecast_num=21,
              data_range=None,
              save_result=False,
              save_shifted_result=False,
              append_info=None):

        x_train, y, x_final = self._train_valid_split(
            train_start=data_range['train_start'],
            valid_start=data_range['valid_start'],
            valid_end=data_range['valid_end'])
        # print(x_train)
        x_valid, y_valid, head_valid = \
            self.series_to_features(x_train, feature_num)
        # print(x_valid, y_valid, head_valid)
        pred_valid = MachineLearningModel(
            x_valid, y_valid, head_valid,
            forecast_num=len(y)).predict(model_name)
        x_final, y_final, head_final = \
            self.series_to_features(x_final, feature_num)
        # print(x_final, y_final, head_final)
        pred_final = MachineLearningModel(
            x_final, y_final, head_final,
            forecast_num=forecast_num).predict(model_name)
        cost = self._calc_acc(y, pred_valid)

        # Save shifted results for different fill modes
        if (self.fill_mode is not None) and save_shifted_result:
            self.get_shifted_results(
                model_name, x_final, y_final, head_final, append_info)

        # Save final results
        if save_result:
            self.save_result(pred_final, model_name, append_info)

        return pred_final, cost, pred_valid


if __name__ == '__main__':

    start_time = time.time()
    utils.check_dir([cfg.result_path])
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
            model_i, feature_num=50, forecast_num=21,
            data_range=range_1, save_result=True)
        print('Model {} Done! Using time {:.2f}s...'.format(
            model_i, time.time() - model_start_time))

    df.to_csv(join(cfg.log_path, 'result_day.csv'))
    utils.thick_line()
    print('All Done! Using time {:.2f}s...'.format(
        time.time() - start_time))
    utils.thick_line()
