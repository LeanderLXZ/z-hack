import utils
import pandas as pd
from os.path import join
from models import TimeSeriesModel


class Training(object):

    def __init__(self):
        pass

    @staticmethod
    def train(mode):

        if mode == 'time_series':
            pred = TimeSeriesModel('day', 'CONTPRICE_DAY')
        else:
            raise ValueError('Wrong Mode!')


if __name__ == '__main__':

    result_path = '../results'
    utils.check_dir([result_path])

    df = pd.DataFrame()

    pred, pred_np = TimeSeriesModel(
        'day', 'CONTPRICE_DAY').predict('arima', forest_num=20, frequency=5)
    # print(pred, pred_np)
    df['arima_day'] = pred_np

    pred, pred_np = TimeSeriesModel(
        'day', 'CONTPRICE_DAY').predict('stl', forest_num=20, frequency=5)
    # print(pred, pred_np)
    df['stl_day'] = pred_np

    pred, pred_np = TimeSeriesModel(
        'day', 'CONTPRICE_DAY').predict('ets', forest_num=20, frequency=5)
    # print(pred, pred_np)
    df['ets_day'] = pred_np

    pred, pred_np = TimeSeriesModel(
        'day', 'CONTPRICE_DAY').predict('hw', forest_num=20, frequency=10)
    # print(pred, pred_np)
    df['hw_day'] = pred_np

    df.to_csv(join(result_path, 'result_day.csv'))
