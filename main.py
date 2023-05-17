from src.econometrics import TimeSeriesModels
import matplotlib.pyplot as plt
import numpy as np


def main():
    pathToData = "data/TSLA.csv"
    myTimeSeriesModel = TimeSeriesModels(pathToData=pathToData)
    myTimeSeriesModel.processData()
    
    # Set what data to be analysed
    inputData = myTimeSeriesModel.returnsClosing
    datesInputData = myTimeSeriesModel.returnClosingDates
    
    # Split data into train and test
    predictionRange = 5
    myTimeSeriesModel.setIndicesTrainTestSplit(len(inputData), predictionRange)
    xTrain, xTest = myTimeSeriesModel.trainTestSplit(inputData)
    
    # Trend estimation of Equity Closing
    equityClose = myTimeSeriesModel.equityClosing.to_numpy()
    trend, residuals = myTimeSeriesModel.trendEstimationMA(equityClose)
    
    # Arima model
    # Orders set to what's been generated from our statistical analysis
    arimaForecast, arimaRmse = myTimeSeriesModel.linearModel(xTrain, xTest, p=2, d=0, q=2)
    
    # GARCH
    garchForecast, garchVolatility = myTimeSeriesModel.GARCH(xTrain, xTest, p=2, q=2)

    # ARIMA-GARCH
    arimaGarchForecastTDist, arimaGarchRmseTDist = myTimeSeriesModel.ARMA_GARCH(xTrain, xTest, pArima=1, dArima=1, qArima=1, pGarch=2, qGarch=2, distribution="t")
    arimaGarchForecastNormDist, arimaGarchRmseNormDist = myTimeSeriesModel.ARMA_GARCH(xTrain, xTest, pArima=1, dArima=1, qArima=1, pGarch=2, qGarch=2, distribution="norm")

    # ------------------------------------------
    # Plots
    # ------------------------------------------
    # Trend Estimation
    fig0, ax0 = plt.subplots(2,1, sharex=False)
    ax0[0].plot(equityClose, label="Equity Close")
    ax0[0].plot(trend, label="Trend Estimation")
    ax0[0].set_title("Estimated trend")
    ax0[0].set_xlabel("Dates")
    ax0[0].set_ylabel("Stock Price")
    ax0[0].grid()
    ax0[0].legend()
    ax0[1].plot(residuals, label="Residuals")
    ax0[1].set_title("Residuals")
    ax0[1].set_xlabel("Dates")
    ax0[1].set_ylabel("Residuals")
    ax0[1].grid()
    ax0[1].legend()
    fig0.tight_layout()

    # ARIMA
    plt.figure()
    plt.plot(datesInputData[myTimeSeriesModel.numberTrainData:], arimaForecast, label='Forecast')
    plt.plot(datesInputData[myTimeSeriesModel.numberTrainData:], xTest, label='Test data')
    plt.title("Forecast - ARIMA")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    
    # GARCH
    indicesXAxis = np.linspace(0, len(inputData)-1, 5, dtype=int)
    fig2, ax2 = plt.subplots(2,1, sharex=False)
    ax2[0].plot(datesInputData[myTimeSeriesModel.numberTrainData:], garchForecast.variance.iloc[-1], color="orange", label='Forecast')
    ax2[0].set_title("Volatility Forecast - Garch")
    ax2[0].grid()
    ax2[0].legend()
    ax2[1].plot(datesInputData[:myTimeSeriesModel.numberTrainData], garchVolatility, label="Volatility")
    ax2[1].plot(datesInputData[myTimeSeriesModel.numberTrainData:], garchForecast.variance.iloc[-1], color="orange", label='Forecast')
    ax2[1].set_xticks(datesInputData[indicesXAxis])
    ax2[1].set_title("Garch process")
    ax2[1].grid()
    ax2[1].legend()
    fig2.tight_layout()

  
   # ARIMA-GARCH
    plt.figure()
    plt.grid()
    plt.title("Forecast ARMA-GARCH")
    plt.plot(datesInputData[myTimeSeriesModel.numberTrainData:], xTest, label='Test data')
    plt.plot(datesInputData[myTimeSeriesModel.numberTrainData:], arimaGarchForecastTDist, label='Forecast, t-distribution')
    plt.plot(datesInputData[myTimeSeriesModel.numberTrainData:], arimaGarchForecastNormDist, label='Forecast, normal distribution',)
    plt.legend()
    plt.tight_layout()

    # Show all plots
    plt.show()
    return


if __name__ == '__main__':
    main()