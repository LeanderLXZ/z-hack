from restoreModels import *
import math
import datetime
import time
import pandas as pd

class Restore():
    def __init__(self):
        self.totalData = pd.DataFrame()
        self.dataNew = pd.DataFrame()

    @staticmethod
    def customRound(x):
        a, b = math.modf(x)
        if b >= 0:
            if a >= 0.5:
                return b + 1
            else:
                return b
        else:
            if a <= -0.5:
                return b - 1
            else:
                return b

    #最终的模型复现部分
    def restoreTotal(self):
        print("=================================================================")
        print("            全行业模型训练结果复现开始，预计需要二十分钟")
        print("=================================================================")
        indexData = pd.read_excel("../data/indexFile/testIndex.xlsx")
        indexData["模型1结果(当季)"] = RestoreTimeSeries().specialModel()
        for index in range(2, 19):
            indexData["模型{}结果(当季)".format(index)] = RestoreOldModels(index).restore()
        indexData["模型19结果(当季)"] = (indexData["模型1结果(当季)"] + indexData["模型11结果(当季)"]) / 2
        for index in range(20, 26):
            indexData["模型{}结果(当季)".format(index)] = RestoreNewModels(index).restore()
        timeModel = ["HoltWinters", "STL", "ETS"]
        timeDict = {"Arima": 29, "HoltWinters": 30, "STL": 31, "ETS": 32}
        for model in timeModel:
            indexData["模型{}结果(当季)".format(timeDict[model])] = RestoreTimeSeries().timeModel(flag=model)
        print("=================================================================")
        print("                  全行业模型训练结果复现结束")
        print("=================================================================")
        self.totalData = indexData

    #最终的模型融合部分
    def computeTotalData(self):
        print("=================================================================")
        print("                     分行业训练结果复现开始")
        print("=================================================================")
        header = pd.read_excel("../data/finalProcess/header.xlsx")
        arrayStatistic = pd.read_pickle("../data/finalProcess/tickerRatio.p")
        dataStatistic = pd.DataFrame(data=arrayStatistic, columns=header.columns)
        indexData = pd.read_excel("../data/indexFile/testIndex.xlsx")
        dataStatistic = pd.merge(indexData, dataStatistic, on="TICKER_SYMBOL", how="left")
        dataTemp = pd.DataFrame()
        self.dataNew = indexData.copy()
        columns = self.totalData.columns[self.totalData.columns.str.contains("结果")]

        for column in columns:
            dataTemp[column] = dataStatistic[column] * self.totalData[column]
        series = dataTemp.sum(axis=1, min_count=1)
        self.dataNew["predict"] = series
        print("=================================================================")
        print("                     分行业训练结果复现结束")
        print("=================================================================")

    #最终的结果生成，采用了自定义的四舍五入
    def console(self):
        self.restoreTotal()
        self.computeTotalData()
        print("=================================================================")
        print("                       合表生成最终结果中")
        print("=================================================================")
        mergeData = pd.read_csv("../data/indexFile/submitIndex.csv")
        timeNow = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        mergeData = pd.merge(mergeData, self.dataNew, on="TICKER_SYMBOL", how="left")
        mergeData["predict"] = mergeData["predict"] + mergeData["previous"]
        mergeData["predict"] = (mergeData["predict"] / 10000).apply(Restore.customRound) / 100
        submitFile = mergeData[["ticker", "predict"]]
        submitFile.to_csv("../submit/submit_{}.csv".format(timeNow), header=False, index=False)

if  __name__ == "__main__":
    print("=================================================================")
    print("                 Alassea lome团队复赛代码开始运行\n"
          "    衷心感谢天池赛委会和通联数据对比赛之中的多次提问给予的及时耐心解答")
    print("=================================================================")
    Restore().console()
    print("=================================================================")
    print("            Alassea lome团队复赛代码运行结束，结果已生成")
    print("=================================================================")




