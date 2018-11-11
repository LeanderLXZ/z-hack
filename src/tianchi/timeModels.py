import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr


#这是时序模型的模块
class R_Models(object):

    def __init__(self, data_series, forest_num=1):
        utils = importr('forecast')
        self.data_series = data_series
        self.forest_num = forest_num

    def data_process(self):
        data_series = robjects.IntVector(self.data_series)
        result = robjects.r['ts'](data_series, frequency=4)
        return result


class Arima(R_Models):

    def auto_arima(self):
        data = self.data_process()
        try:
            model = robjects.r['auto.arima'](data)
            result = robjects.r.forecast(model, self.forest_num)
            return result.rx(4)
        except Exception as e:
            print("Error:", e)
            return np.nan


class STL_ETS(R_Models):

    def stl(self):
        data = self.data_process()
        try:
            result = robjects.r.stlf(data, self.forest_num)
            return result.rx(2).rx2(1)
        except Exception as e:
            print("Error:", e)
            return np.nan

    def ets(self):
        data = self.data_process()
        try:
            model = robjects.r.ets(data)
            result = robjects.r.predict(model, h=self.forest_num)
            return result.rx(2).rx2(1)
        except Exception as e:
            print("Error:", e)
            return np.nan


class HoltWinters(R_Models):

    def hw_mul(self):
        data = self.data_process()
        try:
            model = robjects.r.hw(data)
            result = robjects.r.predict(
                model, h=self.forest_num, seasonal='multiplicative')
            return result.rx(2).rx2(1)
        except Exception as e:
            print("Error:", e)
            return np.nan
