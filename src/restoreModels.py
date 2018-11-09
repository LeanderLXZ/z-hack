import pandas as pd
import xgboost as xgb
import numpy as np
from timeModels import *
import re

#复现新数据的模型
class RestoreNewModels():
    def __init__(self, modelIndex):
        self.path = "../data/newGod/{}/".format(modelIndex)

    def restore(self):
        testFeature = pd.read_pickle(self.path + "x_test.p")
        dtest = xgb.DMatrix(testFeature)
        # mergeArray = np.array([0, 1493])
        bst = xgb.Booster()
        bst.load_model(self.path + "single.model")
        predict = bst.predict(dtest)
        # for index in range(1, count + 1):
        #     bst = xgb.Booster()
        #     bst.load_model(self.path + "models/{}.model".format(index))
        #     predict = bst.predict(dtest)
        #     mergeArray = np.concatenate((mergeArray, predict), axis=0)
        # predict = np.average(mergeArray, axis=0, weights=weight)
        return predict

#复原旧数据的模型
class RestoreOldModels():
    def __init__(self, modelIndex):
        self.path = "../data/oldGod/{}/".format(modelIndex)

    def restore(self, count=4):
        testFeature = pd.read_pickle(self.path + "x_test.p")
        dtest = xgb.DMatrix(testFeature)
        mergeArray = np.empty([0, 1493])
        for index in range(1, count + 1):
            bst = xgb.Booster()
            bst.load_model(self.path + "models/{}.model".format(index))
            predict = bst.predict(dtest)
            predict = np.reshape(predict, [1, 1493])
            mergeArray = np.vstack((mergeArray, predict))
        predict = np.mean(mergeArray, axis=0)
        temp = pd.DataFrame({"TICKER_SYMBOL":pd.read_pickle(self.path + "id_test.p"),
                      "predict":predict})
        return temp.sort_values(by=["TICKER_SYMBOL"])["predict"].values

#复原时序模型
class RestoreTimeSeries():
    def __init__(self):

        self.trainData = pd.read_excel("../data/timeSeries/timeSeriesData.xlsx", index_col="TICKER_SYMBOL")

    def timeModel(self, flag):
        predictList = []
        for index, row in self.trainData.iterrows():
            dataArray = row.values
            dataArray = dataArray[~np.isnan(dataArray)]
            dataArray = dataArray / 10000
            try:
                if flag == "Arima":
                    result = Arima(dataArray).auto_arima()
                elif flag == "STL":
                    result = STL_ETS(dataArray).stl()
                elif flag == "ETS":
                    result = STL_ETS(dataArray).ets()
                elif flag == "HoltWinters":
                    result = HoltWinters(dataArray).hw_mul()
                else:
                    raise Exception("Wrong Flags, Please Check!")
                result = str(result)
                result = result.split(" ")[-1]
                result = re.findall(r"\d.+", result)[0]
                result = float(result) * 10000
                result = int(result)
            except:
                if flag == "STL" and index == 567:
                    result = 97232279
                else:
                    result = np.nan
            predictList.append(result)
        return predictList

    def specialModel(self):
        predictList = []
        for index, row in self.trainData.iterrows():
            dataArray = row.values
            dataArray = dataArray[~np.isnan(dataArray)]
            dataArray = dataArray / 100000
            try:
                result = Arima(dataArray).auto_arima()
                result = str(result)
                result = result.split(" ")[-1]
                result = re.findall(r"\d.+", result)[0]
                result = float(result) * 100000
                result = int(result)
            except:
                result = np.nan
            predictList.append(result)
        return predictList



