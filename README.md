# Generating-Synthetic-Prices-Data

## Overview
This project explores the development of synthetic financial data generation techniques for applications in risk management, specifically stress testing, backtesting trading strategies, and evaluating risk metrics like Expected Shortfall (ES). Synthetic data is crucial for overcoming the limitations of scarce historical datasets and enabling robust model testing under diverse market conditions.

[Access Project Code](https://colab.research.google.com/drive/1gstUw9FwKLSupvSCj6zTBJtORKYD8k5M?usp=sharing)

<img src="ss/Price Plot.png" alt="Stat comp" style="width: 90%;">

## Objectives
The primary objectives of this project include:
1. **Data Preparation**: Cleaning and transforming raw financial data into log returns.
2. **Synthetic Data Generation**: Developing models to generate synthetic return data that replicates real-world statistical properties.
3. **Evaluation and Validation**: Assessing the fidelity of synthetic data using statistical metrics such as mean, variance, skewness, kurtosis, autocorrelation, and Kolmogorov-Smirnov statistics.

## Dataset
- **Asset**: Bitcoin (BTC) hourly price data.
- **Period**: January 2020 to November 2024 (37,080 rows).
- **Transformation**: 
  - Log returns were used to ensure stationarity, scale invariance, and additive properties over time.
  - Returns typically follow a distribution closer to normality, simplifying modeling efforts.

## Methodology

### Data Preparation

1. **Data Cleaning**: Historical price data underwent rigorous cleaning procedures to remove outliers, missing values, and any irregularities that could introduce bias.
2. **Transformation to Log Returns**:
   - The transformation from prices \( P_t \) to log returns \( X_t \) is defined as:
     \[
     X_t = \log\left(\frac{P_t}{P_{t-1}}\right).
     \]
   - Using log returns enhances the stationarity of the time series and simplifies the handling of multiplicative effects.

### Synthetic Data Generation Techniques

#### 1. Traditional Statistical Models

- **Autoregressive (AR)**:
  - The AR model captures linear dependencies by expressing the current value \( X_t \) as a weighted sum of its past values:
    \[
    X_t = c + \phi_1 X_{t-1} + \epsilon_t.
    \]
  - Stationarity was confirmed using the Augmented Dickey-Fuller (ADF) test, which returned a highly negative test statistic (e.g., \(-32.75\)) and a \( p \)-value near 0.0, indicating strong evidence against the presence of a unit root.

- **ARIMA (Autoregressive Integrated Moving Average)**:
  - ARIMA models combine autoregressive and moving average components to address both linear dependencies and shocks:
    \[
    X_t = c + \phi_1 X_{t-1} + \theta_1 \epsilon_{t-1} + \epsilon_t.
    \]
  - Applying ARIMA(1,0,1) to the stationary log returns, residual diagnostics indicated that the residuals approximated white noise, confirming model adequacy for generating synthetic returns.

#### 2. Volatility-Based Models

- **GARCH (Generalized Autoregressive Conditional Heteroskedasticity)**:
  - GARCH models the conditional variance \(\sigma_t^2\) of the time series to capture volatility clustering:
    \[
    \sigma_t^2 = \omega + \alpha_1 \epsilon_{t-1}^2 + \beta_1 \sigma_{t-1}^2.
    \]

- **GARCH with Student-t Distribution**:
  - Incorporating a Student-t distribution for the residuals accounts for heavier tails and extreme market events. Parameter estimation confirmed significant volatility persistence and better alignment with observed return distributions.

- **GARCH with LSTM Hybrid**:
  - An LSTM layer was integrated to model non-linear, complex temporal patterns in volatility dynamics.
  - Training on sequences of past conditional volatilities improved predictive accuracy, resulting in more realistic synthetic volatility patterns.

- **GJR-GARCH (Glosten-Jagannathan-Runkle GARCH)**:
  - GJR-GARCH introduces an asymmetry term to capture leverage effects, where negative shocks increase volatility more than positive shocks of the same magnitude:
    \[
    \sigma_t^2 = \omega + \alpha_1 \epsilon_{t-1}^2 + \gamma I_{\{\epsilon_{t-1}<0\}}\epsilon_{t-1}^2 + \beta_1 \sigma_{t-1}^2.
    \]

#### 3. Fourier Transform Models

- **Fourier Transform with Randomized Phases**:
  - Applying the Fast Fourier Transform (FFT) decomposes the time series into frequency components.
  - Randomizing the phase angles while preserving the power spectrum ensures that the synthetic data retain similar frequency-domain properties but differ in the time domain.
  - Inverse FFT reconstruction yields synthetic series with identical spectral characteristics yet novel temporal sequences.

#### **4. Combined Ensemble Model**
   - Merged outputs from:
     - **GARCH with LSTM**
     - **GARCH with Student-t Distribution**
     - **GJR-GARCH**
   - Synthetic returns were averaged with equal weights to combine strengths:
     - **GARCH**: Captured volatility clustering.
     - **Student-t**: Modeled heavy tails for extreme price movements.
     - **LSTM**: Identified complex non-linear temporal patterns.
   - Ensemble model effectively balanced overfitting and realism, producing highly accurate synthetic data.

## Evaluation Metrics
1. **Moments Analysis**:
   - Compared mean, variance, skewness, and kurtosis of synthetic vs. real data.
2. **Autocorrelation**:
   - Captured temporal dependencies using autocorrelation plots.
3. **Kolmogorov-Smirnov (KS) Test**:
   - Assessed the maximum difference between cumulative distributions of real and synthetic data.
4. **Visual Comparisons**:
   - Plotted real vs. synthetic returns and prices for qualitative validation.
  
<img src="ss/Stat comparision.png" alt="Stat comp" style="width: 90%;">

## Key Insights
1. **Ensemble Models**:
   - Showed the best performance across all statistical metrics.
   - Balanced overfitting while preserving key market dynamics.
2. **Fourier Transform**:
   - Maintained frequency-domain properties but lacked flexibility in capturing distribution nuances.
3. **Volatility Models**:
   - Accurately replicated volatility clustering and heavy tails, critical for stress testing.
4. **Traditional Models**:
   - ARIMA models performed well for capturing linear patterns but struggled with higher-order properties.
  
<div style="display: flex; justify-content: space-between;">
  <img src="ss/KS Stat.png" alt="Game Screenshot 1" style="width: 60%;">
</div>

## Results
The ensemble model, leveraging strengths from statistical, machine learning, and Fourier-based techniques, produced synthetic datasets that closely matched real data's statistical and temporal characteristics. These datasets are suitable for:
- Stress testing under extreme market conditions.
- Backtesting trading strategies with realistic data.
- Risk metric evaluation for robust financial modeling.

<img src="ss/Price Plot - Avg.png" alt="Game Screenshot 2" style="width: 90%;">

## Future Recommendations
1. **Advanced Machine Learning Models**:
   - Explore TimeGAN and Variational Autoencoders (VAEs) for improved sequence generation.
2. **Integrating Exogenous Shocks**:
   - Include macroeconomic variables and news events for scenario-specific stress testing.
3. **Fourier-Deep Learning Hybrid**:
   - Combine Fourier-based methods with deep learning to enhance spectral and temporal synthesis.
