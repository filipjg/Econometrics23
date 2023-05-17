import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")  # setting ignore as a parameter


class DataLoader:
    dateString = "Date"
    openString = "Open"
    highString = "High"
    lowString = "Low"
    closingString = "Close"
    adjClosingString = "Adj Close"
    volumeString = "Volume"

    def __init__(self, pathToData):
        self.pathToData = pathToData

    def readCsv(self):
        # Read csv
        equityData = pd.read_csv(self.pathToData)
        equityData, _ = self.removeNans(equityData)
        self.columns = list(equityData.columns)
        self.numberObservations = equityData.shape[0]
        # Separate DataFrame
        self.equityDates = equityData[self.dateString]
        self.equityOpens = equityData[self.openString]
        self.equityHighs = equityData[self.highString]
        self.equityLows = equityData[self.lowString]
        self.equityClosing = equityData[self.closingString]
        self.equityAdjClosing = equityData[self.adjClosingString]
        self.equityVolumes = equityData[self.volumeString]
        return

    def computeStockReturns(self):
        """
        Compute stock returns and log returns
        """
        self.returnsClosing, indicesReturnClosingDates = self.removeNans(
            self.equityClosing.pct_change(1))
        self.returnClosingDates = self.synchronizeDates(indicesReturnClosingDates)
        self.returnsOpen, indicesReturnOpenDates = self.removeNans(
            self.equityOpens.pct_change(1))
        self.returnOpenDates = self.synchronizeDates(indicesReturnOpenDates)

        logReturns = []
        for i in range(1, self.numberObservations):
            logReturn = np.log(self.equityClosing[i] / self.equityClosing[i - 1])
            logReturns.append(logReturn)
        
        self.logReturn, indicesLogReturnDates = self.removeNans(pd.DataFrame(logReturns, columns=["Log Returns"]))
        self.logReturnDates = self.synchronizeDates(indicesLogReturnDates)
        return

    def removeNans(self, df: pd.DataFrame):
        """Remove Nans from data 

        Args:
            df (pd.DataFrame): input data of choice

        Returns:
            (dfNewIndices, indicesDates): sorted dataframe without Nans, indices without Nans
        """
        dfWithoutNans = df.replace([-np.inf, np.inf], np.nan).dropna()
        indicesDates = dfWithoutNans.index
        dfNewIndices = dfWithoutNans.reset_index(drop=True)
        return dfNewIndices, indicesDates

    def synchronizeDates(self, indicesDates):
        return self.equityDates[indicesDates].reset_index(drop=True)
        
    def computeDifference(self, df: pd.DataFrame):
        dfDiscreteDifferenceElements = df.diff(periods=1)
        return dfDiscreteDifferenceElements

    def processData(self):
        self.readCsv()
        self.computeStockReturns()

    def setIndicesTrainTestSplit(self, numberDataPoints, predictionRange=0, ratioTrainData=0.95):
        self.numberDataPoints = numberDataPoints
        # For Time series prediction range is a small number
        if predictionRange:
            self.predictionRange = predictionRange
            self.numberTrainData = int(numberDataPoints - predictionRange)
            return
        # In case of ratio based splitting
        self.numberTrainData = int(numberDataPoints * ratioTrainData)
        self.predictionRange = numberDataPoints - self.numberTrainData
        return

    def trainTestSplit(self, inputData = pd.DataFrame):
        trainData = inputData[:self.numberTrainData]
        testData = inputData[self.numberTrainData:]
        return trainData, testData
        


if __name__ == "__main__":
    pathToData = "data/TSLA.csv"
    myDataLoader = DataLoader(pathToData)
    myDataLoader.processData()

    # Split data into train and test
    inputData = myDataLoader.returnsClosing
    datesInputData = myDataLoader.returnClosingDates
    myDataLoader.setIndicesTrainTestSplit(len(inputData), 10)
    returnClosingTrain, returnClosingTest = myDataLoader.trainTestSplit(inputData)
    # To only plot few dates
    indicesXAxis = np.linspace(0,len(datesInputData)-1, 10, dtype=int)
    plt.plot(datesInputData, inputData, label="Return Closings")
    plt.plot(datesInputData[:myDataLoader.numberTrainData],returnClosingTrain,":", label="Train")
    plt.plot(datesInputData[myDataLoader.numberTrainData:],returnClosingTest,":", label="Test")
    plt.xticks(datesInputData[indicesXAxis])
    plt.grid()
    plt.legend()
    plt.show()
    
