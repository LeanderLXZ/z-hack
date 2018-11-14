import numpy as np
from rpy2 import robjects as ro


class RModelBase(object):

    def __init__(self, data_series, freq, forest_num):
        ro.r('library(forecast)')

        self.data_series = data_series
        self.frequency = freq
        self.forest_num = forest_num

    def data_process(self):
        data_series = ro.IntVector(self.data_series)
        result = ro.r['ts'](data_series, frequency=self.frequency)
        return result


class Arima(RModelBase):

    def predict(self):
        data = self.data_process()
        try:
            model = ro.r['auto.arima'](data)
            result = ro.r.forecast(model, self.forest_num)
            result_np = np.array(result.rx(4)).reshape(-1)
            return result, result_np
        except Exception as e:
            print('[Error in Arima] ', e)
            return None, np.zeros(self.forest_num)


class STL(RModelBase):

    def predict(self):
        data = self.data_process()
        try:
            result = ro.r.stlf(data, self.forest_num)
            result_np = np.array(result.rx(2)).reshape(-1)
            return result, result_np
        except Exception as e:
            print('[Error in STL] ', e)
            return None, np.zeros(self.forest_num)


class ETS(RModelBase):

    def predict(self):
        data = self.data_process()
        try:
            model = ro.r.ets(data)
            result = ro.r.predict(model, h=self.forest_num)
            result_np = np.array(result.rx(2)).reshape(-1)
            return result, result_np
        except Exception as e:
            print('[Error in ETS] ', e)
            return None, np.zeros(self.forest_num)


class HoltWinters(RModelBase):

    def predict(self, seasonal='multiplicative'):
        data = self.data_process()
        try:
            model = ro.r.hw(data)
            result = ro.r.predict(
                model, h=self.forest_num, seasonal=seasonal)
            result_np = np.array(result.rx(2)).reshape(-1)
            return result, result_np
        except Exception as e:
            print('[Error in HoltWinters] ', e)
            return None, np.zeros(self.forest_num)
