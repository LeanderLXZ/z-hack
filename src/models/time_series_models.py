from models.r_models import Arima, STL, ETS, HoltWinters


class TimeSeriesModel(object):

    def __init__(self, data, freq, forest_num, seasonal=None):
        self.data = data
        self.freq = freq
        self.forest_num = forest_num
        self.seasonal = seasonal

    def predict(self, model_name):

        if model_name == 'arima':
            _, result_np = Arima(self.data, self.freq, self.forest_num).predict()
        elif model_name == 'stl':
            _, result_np = STL(self.data, self.freq, self.forest_num).predict()
        elif model_name == 'ets':
            _, result_np = ETS(self.data, self.freq, self.forest_num).predict()
        elif model_name == 'hw':
            _, result_np = HoltWinters(
                self.data, self.freq, self.forest_num).predict(self.seasonal)
        else:
            raise Exception("Wrong Model Name!")

        return result_np
