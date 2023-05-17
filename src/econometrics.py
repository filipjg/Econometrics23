from src.dataLoader import DataLoader
import numpy as np
import numpy.random as npr
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings('ignore') # setting ignore as a parameter

class TimeSeriesModels(DataLoader):
    
    def __init__(self, pathToData):
        super().__init__(pathToData)
     
    def trendEstimationMA(self, inputData:np.ndarray):
        """Trend estimation Moving Average

        Args:
            inputData (np.ndarray): input of choice

        Returns:
            (trendEstimation, residuals)
        """
        
        movingAverageLag = int(1)
        doubledMovingAverageLag = 2 * movingAverageLag
        numberTrendData = int(len(inputData) - doubledMovingAverageLag)
        trendEstimation = np.zeros((numberTrendData))
        residuals = np.zeros_like(trendEstimation)
        for i in range(doubledMovingAverageLag, numberTrendData):
            trendEstimation[i] = 1/(1+doubledMovingAverageLag)*np.sum([inputData[i-1:i+doubledMovingAverageLag]])
            residuals[i] = inputData[i] - trendEstimation[i]
        return trendEstimation, residuals

    def linearModel(self, xTrain, xTest, p=2, d=0, q=2):
        """Fit and predict ARIMA

        Args:
            xTrain (pd.Dataframe): Training data
            xTest (pd.Dataframe): Test data
            p (int, optional): order of autoregressive model. Defaults to 2.
            d (int, optional): order of derivatives. Defaults to 0.
            q (int, optional): order of moving average model. Defaults to 2.

        Returns:
            (arimaForecast, rmse)
        """
        arimaModel = ARIMA(xTrain, order=(p,d,q))
        arimaFitted = arimaModel.fit()

        arimaForecast = arimaFitted.forecast(len(xTest)) 
        arimaPredict = arimaFitted.predict(start=len(xTrain)+1, end=len(xTrain)+len(xTest))

        rmse = mean_squared_error(xTest, arimaPredict)
        print("ARIMA RMSE", rmse) 
        return arimaForecast, rmse
    
    def GARCH(self, xTrain, xTest, p=2, q=2):
        """Fit and predict Garch

        Args:
            xTrain (pd.Dataframe): Training data
            p (int, optional): order of autoregressive model. Defaults to 2.
            q (int, optional): order of moving average model. Defaults to 2.

        Returns:
            (garchForecast, garchVolatility)
        """
        
        garchModel = arch_model(xTrain,vol='GARCH',p=p,q=q)
        garchResults = garchModel.fit(update_freq = 5)
        garchVolatility = garchResults.conditional_volatility**2
        garchForecast = garchResults.forecast(horizon=len(xTest))
        return garchForecast, garchVolatility
    
    def ARMA_GARCH(self, xTrain, xTest, pArima=1, dArima=1, qArima=1, pGarch=2, qGarch=2, distribution="t"):
        """Fit and predict ARIMA-Garch

        Args:
            xTrain (pd.Dataframe): Training data
            xTest (pd.Dataframe): Test data
            pArima (int, optional): order of autoregressive model. Defaults to 1.
            dArima (int, optional): order of derivatives. Defaults to 1.
            qArima (int, optional): order of moving average model. Defaults to 1.
            pGarch (int, optional): order of autoregressive model. Defaults to 2.
            qGarch (int, optional): order of moving average model. Defaults to 2.
            distribution (str): t, norm, Defaults to t.

        Returns:
            (equityForecast, rmse)
        """

        equityForecast = []
        equityExtension = list(xTrain)
        
        """ Multiple time steps """
        for i in range(len(xTest)):
            arimaModel = ARIMA(equityExtension, order=(pArima,dArima,qArima))
            arimaFitted = arimaModel.fit()
            arimaResiduals = arimaFitted.resid
            
            garchModel = arch_model(arimaResiduals,vol='GARCH',p=pGarch,q=qGarch)
            garchFitted = garchModel.fit()

            predictedMean = arimaFitted.predict(n_periods=1)[-1]
            garchForecast = garchFitted.forecast(horizon=1)
            predictedGarchComponent = garchForecast.variance['h.1'].iloc[-1]
            
            if distribution == "t":
                predictedARIMA = predictedMean + np.sqrt(predictedGarchComponent)*npr.standard_t(2)
            elif distribution == "norm":
                predictedARIMA = predictedMean + np.sqrt(predictedGarchComponent)*npr.normal(0,0.05)
            else:
                print("Please use supported distributions: [t, norm]")

            equityExtension.append(predictedARIMA)
            equityForecast.append(predictedARIMA)

        rmse = mean_squared_error(xTest, equityForecast)
        print("MSE ARMA-GARCH", rmse)
        return equityForecast, rmse