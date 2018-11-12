from models import utils
import preprocess
import pandas as pd
from os.path import join
from models.time_series_models import TimeSeriesModel


class Training(object):

    def __init__(self, sample_mode):
        file_path = preprocess.preprocessed_path
        self.data = pd.read_csv(
            join(file_path, 'total_{}.csv'.format(sample_mode)))
        self.suffix = utils.get_suffix(sample_mode)

    def train(self, model_name, params=None):

        if model_name in ['arima', 'stl', 'ets', 'hw']:
            data_series = self.data['CONTPRICE_{}'.format(self.suffix)].values
            pred = TimeSeriesModel(data_series, params).predict(model_name)
        else:
            raise ValueError('Wrong Mode!')

        return pred


if __name__ == '__main__':

    result_path = '../results'
    utils.check_dir([result_path])

    df = pd.DataFrame()
    T = Training('day')
    parameters = {'frequency': 5,
                  'forest_num': 20}
    df['arima_day'] = T.train('arima', parameters)
    df['stl_day'] = T.train('stl', parameters)
    df['ets_day'] = T.train('ets', parameters)
    parameters = {'frequency': 10,
                  'forest_num': 20,
                  'hw_seasonal': 'multiplicative'}
    df['hw_day'] = T.train('hw', parameters)
    df.to_csv(join(result_path, 'result_ts_day.csv'))
