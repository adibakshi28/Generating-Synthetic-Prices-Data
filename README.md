# Generating-Synthetic-Prices-Data

## Overview
This project explores the development of synthetic financial data generation techniques for applications in risk management, specifically stress testing, backtesting trading strategies, and evaluating risk metrics like Expected Shortfall (ES). Synthetic data is crucial for overcoming the limitations of scarce historical datasets and enabling robust model testing under diverse market conditions.

[Access Project Code](https://colab.research.google.com/drive/1gstUw9FwKLSupvSCj6zTBJtORKYD8k5M?usp=sharing)

---

## Objectives
The primary objectives of this project include:
1. **Data Preparation**: Cleaning and transforming raw financial data into log returns.
2. **Synthetic Data Generation**: Developing models to generate synthetic return data that replicates real-world statistical properties.
3. **Evaluation and Validation**: Assessing the fidelity of synthetic data using statistical metrics such as mean, variance, skewness, kurtosis, autocorrelation, and Kolmogorov-Smirnov statistics.

---

## Dataset
- **Asset**: Bitcoin (BTC) hourly price data.
- **Period**: January 2020 to November 2024 (37,080 rows).
- **Transformation**: 
  - Log returns were used to ensure stationarity, scale invariance, and additive properties over time.
  - Returns typically follow a distribution closer to normality, simplifying modeling efforts.

---

## Methodology
### Data Preparation
1. Raw price data was cleaned to remove inconsistencies.
2. Transformed into log returns:
   - \(\text{Log Return} = \log(\frac{P_t}{P_{t-1}})\)
   - Log returns offer mathematical convenience and stationarity for time-series modeling.

---

### Synthetic Data Generation Techniques

#### **1. Traditional Statistical Models**
   - **Autoregressive (AR)**:
     - Captures linear dependencies in time-series data by modeling the current value as a weighted sum of its past values.
     - Stationarity was confirmed using the Augmented Dickey-Fuller (ADF) test with a highly negative test statistic (\(-32.75\)) and \(p\)-value of 0.0, indicating strong rejection of the null hypothesis of non-stationarity.
   - **ARIMA (Autoregressive Integrated Moving Average)**:
     - Combined autoregressive (\(AR\)) and moving average (\(MA\)) terms to model both linear dependencies and shocks in the data.
     - ARIMA(1, 0, 1) was applied to stationary log returns:
       \[
       X_t = c + \phi_1 X_{t-1} + \theta_1 \epsilon_{t-1} + \epsilon_t
       \]
       where \(\phi_1\) and \(\theta_1\) capture autoregressive and moving average effects, respectively.
     - Residual diagnostics confirmed that residuals are white noise, ensuring the model’s validity for generating synthetic returns.

#### **2. Volatility-Based Models**
   - **GARCH (Generalized Autoregressive Conditional Heteroskedasticity)**:
     - Captures time-varying volatility by estimating conditional variance:
       \[
       \sigma_t^2 = \omega + \alpha_1 \epsilon_{t-1}^2 + \beta_1 \sigma_{t-1}^2
       \]
     - Successfully modeled volatility clustering observed in financial returns.
   - **GARCH with Student-t Distribution**:
     - Enhanced the basic GARCH model to account for heavy-tailed behavior in financial returns by using a Student-t distribution for residuals.
     - Fitted model parameters indicated significant persistence of volatility.
   - **GARCH with LSTM Hybrid**:
     - Combined GARCH with a Long Short-Term Memory (LSTM) network to capture non-linear and temporal patterns in volatility dynamics.
     - LSTM was trained on sequences of past conditional volatilities, improving predictive accuracy.
   - **GJR-GARCH (Glosten-Jagannathan-Runkle GARCH)**:
     - Extended the GARCH framework to include the leverage effect:
       - Negative shocks disproportionately increased volatility compared to positive shocks of the same magnitude.
     - This asymmetry was modeled using an indicator function to adjust the conditional variance.

#### **3. Fourier Transform Models**
   - **Fourier Transform with Randomized Phases**:
     - Preserved the spectral (frequency-domain) properties of log returns while introducing randomness in the time domain.
     - Applied the Fast Fourier Transform (FFT) to decompose the time series into magnitudes and phases.
     - Original phases were replaced with uniformly distributed random phases, preserving power spectrum while disrupting temporal structure.
     - Reconstructed synthetic returns via the Inverse FFT, ensuring the same frequency characteristics as the original data.

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

---

## Evaluation Metrics
1. **Moments Analysis**:
   - Compared mean, variance, skewness, and kurtosis of synthetic vs. real data.
2. **Autocorrelation**:
   - Captured temporal dependencies using autocorrelation plots.
3. **Kolmogorov-Smirnov (KS) Test**:
   - Assessed the maximum difference between cumulative distributions of real and synthetic data.
4. **Visual Comparisons**:
   - Plotted real vs. synthetic returns and prices for qualitative validation.

---

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

---

## Results
The ensemble model, leveraging strengths from statistical, machine learning, and Fourier-based techniques, produced synthetic datasets that closely matched real data's statistical and temporal characteristics. These datasets are suitable for:
- Stress testing under extreme market conditions.
- Backtesting trading strategies with realistic data.
- Risk metric evaluation for robust financial modeling.

---

## Future Recommendations
1. **Advanced Machine Learning Models**:
   - Explore TimeGAN and Variational Autoencoders (VAEs) for improved sequence generation.
2. **Integrating Exogenous Shocks**:
   - Include macroeconomic variables and news events for scenario-specific stress testing.
3. **Fourier-Deep Learning Hybrid**:
   - Combine Fourier-based methods with deep learning to enhance spectral and temporal synthesis.

---
