import pandas as pd
import data_preprocess
from os.path import join
from r_models import Arima, STL, ETS, HoltWinters


class TimeSeriesModel(object):

    def __init__(self, sample_mode, col_name):

        file_path = data_preprocess.preprocessed_path

        self.data = pd.read_csv(
            join(file_path, 'total_{}.csv'.format(sample_mode)),
            usecols=[col_name]).values.reshape(-1)

    def predict(self, model_name, forest_num, frequency):

        if model_name == 'arima':
            result, result_np =\
                Arima(self.data, forest_num, frequency).predict()
        elif model_name == 'stl':
            result, result_np = \
                STL(self.data, forest_num, frequency).predict()
        elif model_name == 'ets':
            result, result_np = \
                ETS(self.data, forest_num, frequency).predict()
        elif model_name == 'hw':
            result, result_np = \
                HoltWinters(self.data, forest_num, frequency).predict()
        else:
            raise Exception("Wrong Flags, Please Check!")

        return result, result_np


if __name__ == '__main__':

    pass
