from models.r_models import Arima, STL, ETS, HoltWinters


class TimeSeriesModel(object):

    def __init__(self, data, params):
        self.data = data
        self.params = params

    def predict(self, model_name):

        if model_name == 'arima':
            _, result_np = Arima(self.data, self.params).predict()
        elif model_name == 'stl':
            _, result_np = STL(self.data, self.params).predict()
        elif model_name == 'ets':
            _, result_np = ETS(self.data, self.params).predict()
        elif model_name == 'hw':
            _, result_np = HoltWinters(self.data, self.params).predict()
        else:
            raise Exception("Wrong Model Name!")

        return result_np
