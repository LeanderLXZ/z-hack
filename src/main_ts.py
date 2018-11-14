from models import utils
import pandas as pd
import numpy as np
from os.path import join
from models.time_series_models import TimeSeriesModel
from config import cfg


class Training(object):

    def __init__(self, sample_mode, select_col):

        self.sample_mode = sample_mode
        self.select_col = select_col
        self.suffix = utils.get_suffix(sample_mode)

        file_path = cfg.preprocessed_path
        self.data = pd.read_csv(
            join(file_path, 'total_{}.csv'.format(sample_mode)))

    def _train_valid_split(self, train_start, valid_start, valid_end):

        if self.sample_mode == 'day':
            train_started = False
            valid_started = False
            valid_ended = False
            train_data = []
            valid_data = []
            final_data = []
            for row_iter in self.data.iterrows():
                row = row_iter[1]
                date_str = '{}-{}-{}'.format(
                    int(row['YEAR']), int(row['MONTH']), int(row['DAY']))
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
    def _calc_acc(y, pred):
        if pred.sum() < 1:
            return np.nan
        assert len(y) == len(pred), (len(y), len(pred))
        cost = []
        for y_i, pred_i in zip(y, pred):
            cost.append(abs(y_i - pred_i))
        return np.mean(cost)

    def train(self, model_name, freq, forest_num=21, seasonal='multiplicative',
              data_range=None, save_result=False, append_info=None):

        if data_range:
            x_train, y, x_final = self._train_valid_split(
                train_start=data_range['train_start'],
                valid_start=data_range['valid_start'],
                valid_end=data_range['valid_end'])
            # print(x_train, y, x_final)
            pred_valid = TimeSeriesModel(
                x_train, freq, forest_num=len(y),
                seasonal=seasonal).predict(model_name)
            pred_final = TimeSeriesModel(
                x_final, freq, forest_num,
                seasonal=seasonal).predict(model_name)
            cost = self._calc_acc(y, pred_valid)

        else:
            x_train = self.data[
                '{}_{}'.format(self.select_col, self.suffix)].values
            pred_final = TimeSeriesModel(
                x_train, freq, forest_num,
                seasonal=seasonal).predict(model_name)
            pred_valid = None
            cost = None

        if save_result:
            utils.check_dir([cfg.result_path])
            df_pred = pd.read_csv(
                join(cfg.source_path, 'z_hack_submit_new.csv'),
                index_col=['FORECASTDATE'], usecols=['FORECASTDATE'])
            assert len(df_pred) == len(pred_final), \
                (len(df_pred), len(pred_final))
            df_pred['FORECASTPRICE'] = pred_final
            if append_info is None:
                append_info = '_' + model_name + '_' + self.sample_mode
            df_pred.to_csv(join(cfg.result_path,
                           'result{}.csv'.format(append_info)),
                           float_format='%.2f')

        return pred_final, cost, pred_valid


if __name__ == '__main__':

    utils.check_dir([cfg.result_path])

    df = pd.read_csv(join(cfg.source_path, 'z_hack_submit_new.csv'),
                     index_col=['FORECASTDATE'], usecols=['FORECASTDATE'])
    T = Training('day', 'CONTPRICE')

    range_1 = {'train_start': '2011-1-4',
               'valid_start': '2013-1-4',
               'valid_end': '2013-1-31'}

    df['arima_day'], _, _ = T.train(
        'arima', freq=5, forest_num=21, data_range=range_1, save_result=True)
    df['stl_day'], _, _ = T.train(
        'stl', freq=5, forest_num=21, data_range=range_1, save_result=True)
    df['ets_day'], _, _ = T.train(
        'ets', freq=5, forest_num=21, data_range=range_1, save_result=True)
    df['hw_day'], _, _ = T.train(
        'hw', freq=10, forest_num=21, seasonal='multiplicative',
        data_range=range_1, save_result=True)

    df.to_csv(join(cfg.log_path, 'result_day.csv'))
