# Stock Price Prediction using LSTM-XGBoost Hybrid Model

This project predicts stock prices using a hybrid model that combines the sequential data modeling capabilities of Long Short-Term Memory (LSTM) networks with XGBoost's ability to handle residual errors. The data is sourced from Yahoo Finance, covering historical stock prices.

---

## 1. Libraries and Modules Import

This section imports the essential libraries required for data preprocessing, model development, and evaluation:
- **`pandas`, `numpy`:** Data manipulation and numerical computations, including array handling and scaling.
- **`matplotlib`:** For data visualization, aiding in understanding trends and comparing results.
- **`tensorflow/keras`:** For implementing LSTM layers in the sequential neural network.
- **`yfinance`:** Fetches real-time and historical stock price data for financial analysis.
- **`xgboost`:** Implements gradient-boosted regression trees to model the residual errors from LSTM predictions.

---

## 2. Data Loading

Stock data is fetched programmatically using the **`yfinance`** library:
- **Input Parameters:**
  - `START`: Start date for fetching data.
  - `TODAY`: Current date to dynamically fetch the latest data.
  - `ticker`: Ticker symbol of the stock (e.g., "RELIANCE.NS").
- **Outputs:**
  - A DataFrame with columns like `Date`, `Open`, `High`, `Low`, `Close`, and `Adj Close`.
- **Importance:** Automates data collection, ensuring the latest stock prices are available.

---

## 3. Data Preprocessing

- **Column Dropping:** Removed `Date` (redundant post-reset index) and `Adj Close` (adjusted price, which is not required for prediction).
- **Purpose:** Focused on `Close` prices, as they reflect the final traded price, crucial for understanding trends.
- **Visualization:** Plotted the `Close` prices to identify patterns, anomalies, or trends over time.

---

## 4. Exponential Moving Averages (EMA)

- **EMA (100 days):** Calculated using exponential smoothing with a span of 100 days to track short-term price trends.
- **EMA (200 days):** Computed to capture long-term trends, providing a broader perspective.
- **Significance:**
  - Highlights underlying trends by smoothing fluctuations.
  - Used in trading strategies to identify support and resistance levels.
- **Visualization:** Plotted EMAs alongside the `Close` prices for comparison.

---

## 5. Data Splitting

- **Training Set:** 70% of the data used for model training to learn historical patterns.
- **Test Set:** Remaining 30% used for evaluating model performance on unseen data.
- **Purpose:** Ensures a fair split to train the model on historical data while testing its ability to generalize.

---

## 6. Data Scaling

- **Scaler Used:** `MinMaxScaler` scales the `Close` price values between 0 and 1.
- **Purpose:**
  - Prevents dominance of larger values during model training.
  - Optimizes the convergence process in gradient-based learning algorithms.
- **Applied To:**
  - Training set (`train_close`) for model training.
  - Test set (`test_close`) for evaluation consistency.

---

## 7. LSTM Model Development

Built a sequential LSTM model to predict time-series data:
- **Input Features:** Past 100 days of scaled `Close` prices.
- **Architecture Details:**
  - **Input Layer:** Accepts a sequence of 100 normalized values.
  - **Four LSTM Layers:**
    - Hidden units: 50, 60, 80, 120 (increasing complexity to capture intricate patterns).
    - **Activation:** ReLU for non-linear transformations.
    - **Dropout:** Gradually increasing rates (0.2 to 0.5) to reduce overfitting.
  - **Output Layer:** A dense layer with 1 neuron to predict a single value (next day's price).
- **Training:**
  - Optimizer: `Adam`, which adapts learning rates for faster convergence.
  - Loss Function: `Mean Squared Error` minimizes the squared difference between actual and predicted values.
  - Epochs: Trained for 10 iterations over the training data.
- **Output:** LSTM predicts the normalized price for the next day.

---

## 8. Testing LSTM Predictions

- **Testing Data Preparation:** Combined the last 100 days of training data with the test set for sequential continuity.
- **Prediction Process:** Generated predicted `Close` prices using the trained LSTM model.
- **Comparison:** Scaled predictions are inverse-transformed to the original scale and compared against actual prices.
- **Visualization:** Plotted actual vs. predicted prices to assess the model's ability to capture stock price movements.

---

## 9. Residual Correction Using XGBoost

XGBoost is used to refine LSTM predictions by modeling residuals:
- **Residual Calculation:** Difference between LSTM-predicted and actual values (errors).
- **Training XGBoost:**
  - **Input:** Residuals as labels and LSTM predictions as features.
  - **Parameters:** 
    - Objective: `reg:squarederror` for regression.
    - Learning Rate: `0.01` to avoid overfitting.
    - Max Depth: `5` for optimal tree complexity.
    - Estimators: `10` for lightweight models.
- **Prediction Correction:** Predicted residuals are added back to the LSTM predictions, improving accuracy.
- **Purpose:** Combines strengths of LSTM (sequential learning) and XGBoost (error modeling) for better performance.

---

## 10. Evaluation

### 10.1. Evaluation (LSTM)

- **Metric Used:** Mean Absolute Error (MAE) to measure prediction accuracy.
- **Result:** Reflects the baseline accuracy of LSTM before residual correction.

### 10.2. Evaluation (XGBoost)

- **Residual Correction Impact:** Evaluated the hybrid model combining LSTM and XGBoost.
- **Metric:** MAE calculated for the corrected predictions.
- **Result:** Demonstrated improved accuracy of the hybrid model, highlighting the benefit of residual correction.
