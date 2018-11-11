import numpy as np
import pandas as pd
from rpy2 import robjects as ro


class RModelBase(object):

    def __init__(self, data_series, forest_num=1, frequency=5):
        ro.r('library(forecast)')

        self.data_series = data_series
        self.forest_num = forest_num
        self.frequency = frequency

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
            print("Error:", e)
            return np.nan


class STL(RModelBase):

    def predict(self):
        data = self.data_process()
        try:
            result = ro.r.stlf(data, self.forest_num)
            result_np = np.array(result.rx(2)).reshape(-1)
            return result, result_np
        except Exception as e:
            print("Error:", e)
            return np.nan


class ETS(RModelBase):

    def predict(self):
        data = self.data_process()
        try:
            model = ro.r.ets(data)
            result = ro.r.predict(model, h=self.forest_num)
            result_np = np.array(result.rx(2)).reshape(-1)
            return result, result_np
        except Exception as e:
            print("Error:", e)
            return np.nan


class HoltWinters(RModelBase):

    def predict(self):
        data = self.data_process()
        try:
            model = ro.r.hw(data)
            result = ro.r.predict(
                model, h=self.forest_num, seasonal='multiplicative')
            result_np = np.array(result.rx(2)).reshape(-1)
            return result, result_np
        except Exception as e:
            print("Error:", e)
            return np.nan
