# Econometrics

## Installation
Create a virual enviroment, activate it and install its requirements. 
```
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Prerequisite
Create a directory to place your downloaded stock data

```
mkdir data
```

## Run code
The file ´dataAnalysis.ipynb´ looks runs code to analyse if the data is stationary, which is to say the time series has:

- Constant mean: there exist $ \mu \in$ $\mathbb{R}$ such that $\mu_{X}$(t) = $\mu$ for all t $\in$ Z
- Constant autocovariance: $\gamma_{X}$(r,s) =  $\gamma_{X}$(r+h,s+h) for all s,r,h $ \in $ Z

The notebook also looks at suitable distributions of the data and model order which can be used when running the models in ´main.py´

Run the different models by

```
python3 main.py
```

## Results
Find resulting plots in dir ´results´