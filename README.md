
**Note: "Insights obtained" section of the README includes only my learnings from project implementation.**

## Table of Contents:
- [About](#about)
- [Screenshots](#screenshots)
- [Insights obtained](#insights-obtained)


## About:
**Description:** needs to add a 90-day prediction in posterior predictive distribution

[Upwork gig link](https://www.upwork.com/jobs/~01e8bd802823c9cd1d)

## Screenshots:

![LSTM model fitting the data](https://github.com/user-attachments/assets/3d74e39c-0714-41bb-a10f-c15ba86bf73b)
**LSTM model fitting the data**


![Evaluation metrics for LSTM](https://github.com/user-attachments/assets/bf068d11-081e-4294-bcfb-b00659a67f86)

**Evaluation metrics for LSTM**


![LSTM 90-day prediction](https://github.com/user-attachments/assets/045f760b-767d-4d8e-8922-812437d3e400)
**LSTM 90-day prediction**

## Insights obtained:
**Note: This section of the README includes only my learnings from project implementation.**

Best 3 sophisticated methods to solve this time series problem are LSTM, GNN, ARCH

**Baseline models:**
Naive Forecasting, Moving Average(MA), Exponential Smoothing (ETS), ARIMA

**Sophisticated models:**
Long Short-Term Memory (LSTM),
Graph Neural Networks (GNN),
Autoregressive Conditional Heteroskedasticity (ARCH)
Kalman filtering and smoothing procedure (https://www.degruyter.com/document/doi/10.1515/em-2018-0005/html?lang=en)
Bayesian models: Bayesian STS, Bayesian DLMs, Bayesian ARIMA, Bayesian GPs, Bayesian NNs, Bayesian State Space model
Seasonal-Trend Decomposition (STL)

**Other models:**
Hybrid Models, Regime-Switching Models, Bayesian Models, Facebook Prophet

**Working with Google dataset:**

[Dataset link](https://www.kaggle.com/competitions/web-traffic-time-series-forecasting/rules)

'!vote_en.wikipedia.org_all-access_all-agents_2017-01-01'

This is an example of a full page name. We need to extract only page name from here excluding "_2017-01-01" to map out the IDs

page name = !vote_en.wikipedia.org_all-access_all-agents

**Working with time series data:**

1) Importing data
2) Visualizing the data with outliers using line chart
3) Detecting outliers using zscore method and removing them
4) Visualizing the data without outliers using line chart
5) Checking whether data is stationary or non-stationary using ADF, PP, KPSS tests
   adf_result = adf_pvalue < 0.05
   pp_result = pp_pvalue < 0.05
   kpss_result = kpss_pvalue > 0.05
6) If data is stationary, d=0 and skip to step: 9) .If data is non-stationary then go to next step
7) Convert the data to stationary using differencing. No of times the data is differenced (d) to bring it to stationary is noted down
8) ADF, PP, KPSS tests are performed to confirm whether the data is stationary
9) p,q values are determined from PACF and ACF plots (where the plot cuts off)
10) The ARIMA model is trained using p,d,q order and model summary is printed
11) Residuals and ACF of residuals is printed. Residuals should hover around y=0 and there should be no patterns.
   In ACF, most autocorrelations must be within confidence interval (except the one at x=0) and should not show significant spikes
12) Actual and predicted data is plotted on single graph to check how well our model is performing
13) MAE, RMSE, MAPE is calculated and these values are compared with other models to find out the best performing one

**Conclusion:**
1. ARIMA, XGBoost models are not giving great results
2. LSTM is giving great results
3. BiLSTM, regularization (l1,l2) shouldn't be used with time series data as they don't give great results

**Example of an ideal LSTM, parameters for time series:**

model = Sequential([
    Input(shape=(SEQ_LENGTH, 1)),
    LSTM(128, return_sequences=True, activation='tanh'),
    Dropout(0.3),
    LSTM(64, return_sequences=False, activation='tanh'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='linear')
])
SEQ_LENGTH = 35
batch_size=32
epochs=1400
