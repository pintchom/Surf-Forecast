# Product Requirements Document: Deep Learning Surf Forecasting System

**Version:** 1.0  
**Date:** October 27, 2025  
**Project Team:** Max Pintchouk, Jason Cho  
**Document Owner:** Project Team  

---

## 1. Executive Summary

### 1.1 Product Vision
Build a deep learning-based time series forecasting system that predicts surf conditions (wave height and period) 1-6 hours into the future using historical NOAA buoy data, with accuracy comparable to or exceeding traditional numerical weather models.

### 1.2 Target Users
- Recreational and professional surfers seeking accurate local surf forecasts
- Surf schools and instructors planning sessions
- Beach safety personnel assessing conditions

### 1.3 Success Criteria
- Achieve RMSE ≤ 0.3m for 1-hour ahead wave height predictions
- Achieve RMSE ≤ 0.5m for 6-hour ahead wave height predictions
- Outperform naive baseline (persistence model) by ≥20% across all time horizons
- Complete model training and evaluation within 8-week timeline

---

## 2. Technical Objectives

### 2.1 Primary Objectives
1. **Time Series Prediction**: Forecast SWH and DWP at three time horizons (1h, 3h, 6h)
2. **Model Comparison**: Evaluate and compare baseline, LSTM, and GRU architectures
3. **Feature Engineering**: Identify optimal input features and lookback windows
4. **Interpretability**: Provide feature importance analysis and error characterization

### 2.2 Secondary Objectives
1. Build reusable data pipeline for multiple buoy stations
2. Create visualization tools for model performance analysis
3. Develop simple web interface for prediction demonstration (stretch goal)

---

## 3. Data Requirements

### 3.1 Data Sources

#### Primary Data: NOAA National Data Buoy Center (NDBC)
- **Station 46012**: Half Moon Bay, CA (37.361°N, 122.879°W)
- **Station 46221**: Santa Barbara, CA (34.274°N, 119.877°W)
- **Access Method**: NDBC HTTP API or bulk download
- **Update Frequency**: Hourly measurements
- **Historical Range**: Minimum 3 years of continuous data required

#### Data Variables (Features)
| Variable | Unit | Description | Required |
|----------|------|-------------|----------|
| Significant Wave Height (WVHT) | meters | SWH - mean of highest 1/3 waves | Yes |
| Dominant Wave Period (DPD) | seconds | Period of most energetic waves | Yes |
| Average Wave Period (APD) | seconds | Mean of all wave periods | No |
| Wave Direction (MWD) | degrees | Direction waves coming from | No |
| Wind Speed (WSPD) | m/s | Sustained wind speed | Yes |
| Wind Direction (WDIR) | degrees | Direction wind coming from | No |
| Wind Gust (GST) | m/s | Peak wind gust | No |
| Atmospheric Pressure (PRES) | hPa | Sea-level pressure | Yes |
| Air Temperature (ATMP) | °C | Air temperature | No |
| Water Temperature (WTMP) | °C | Sea surface temperature | No |

### 3.2 Data Quality Requirements

#### Data Completeness
- Maximum acceptable missing data: 5% per year
- No gaps longer than 6 consecutive hours
- Missing value handling strategy required

#### Data Validation
- Wave height range: 0-20 meters (discard outliers beyond physical limits)
- Wave period range: 1-30 seconds
- Wind speed range: 0-50 m/s
- Pressure range: 950-1050 hPa

#### Temporal Requirements
- Temporal resolution: 1 hour (native buoy sampling)
- Training period: 2018-2023 (5 years minimum)
- Validation period: 2024 Q1-Q2 (6 months)
- Test period: 2024 Q3-Q4 (6 months)

### 3.3 Data Split Strategy
```
Training:   60% (oldest data)
Validation: 20% (middle period)  
Test:       20% (most recent data)
```
**Critical**: Maintain strict temporal ordering - no data leakage from future to past

---

## 4. Feature Engineering Requirements

### 4.1 Input Features

#### Core Features (Required)
- Significant Wave Height (WVHT) - lag sequences
- Dominant Wave Period (DPD) - lag sequences
- Wind Speed (WSPD) - lag sequences
- Atmospheric Pressure (PRES) - lag sequences

#### Derived Features (To Evaluate)
- Wave power: proportional to H² × T (wave height squared × period)
- Pressure gradient: rate of change in atmospheric pressure
- Wind-wave alignment: angle difference between wind and wave direction
- Temporal features: hour of day, day of year (cyclical encoding)
- Rolling statistics: 6-hour, 12-hour, 24-hour moving averages

### 4.2 Lookback Window
- **Initial Configuration**: 24 hours (24 timesteps at 1-hour resolution)
- **Hyperparameter Search Range**: 12, 24, 48, 72 hours
- **Rationale**: Ocean swells can take 12-48 hours to develop; need sufficient historical context

### 4.3 Normalization Strategy
- **Method**: StandardScaler (z-score normalization)
- **Fit on**: Training set only
- **Apply to**: Train, validation, and test sets
- **Per-feature**: Independent scaling for each feature
- **Inverse transform**: Required for interpretable predictions

---

## 5. Model Architecture Requirements

### 5.1 Baseline Models (Week 3-4)

#### Model 1: Persistence Model
- **Description**: Predict t+n = t (naive forecast)
- **Purpose**: Establish lower bound performance
- **Implementation**: Simple value propagation

#### Model 2: Linear Regression
- **Architecture**: Multivariate linear regression
- **Input**: Flattened lookback window features
- **Output**: Multi-horizon predictions
- **Regularization**: L2 (Ridge) with cross-validated alpha

#### Model 3: Feedforward Neural Network
- **Architecture**:
  ```
  Input Layer: [lookback × features]
  Dense(128, relu)
  Dropout(0.2)
  Dense(64, relu)
  Dropout(0.2)
  Dense(32, relu)
  Output Layer: [forecast_horizon × 2]  # SWH and DWP
  ```
- **Loss**: Mean Squared Error (MSE)
- **Optimizer**: Adam (lr=0.001)

### 5.2 LSTM Model (Week 5)

#### Architecture Specifications
```python
Model: Sequential
├── LSTM(units=64, return_sequences=True, input_shape=(lookback, n_features))
├── Dropout(0.2)
├── LSTM(units=32, return_sequences=False)
├── Dropout(0.2)
├── Dense(32, activation='relu')
├── Dense(forecast_horizon * n_targets)
└── Reshape((forecast_horizon, n_targets))
```

#### Hyperparameters to Tune
| Parameter | Search Space | Initial Value |
|-----------|--------------|---------------|
| LSTM layers | [1, 2, 3] | 2 |
| Units per layer | [32, 64, 128, 256] | 64 |
| Dropout rate | [0.1, 0.2, 0.3, 0.4] | 0.2 |
| Lookback window | [12, 24, 48, 72] | 24 |
| Learning rate | [0.0001, 0.001, 0.01] | 0.001 |
| Batch size | [16, 32, 64] | 32 |

#### Training Configuration
- **Loss function**: MSE (primary), MAE (monitored)
- **Optimizer**: Adam with optional learning rate scheduling
- **Early stopping**: Patience=20 epochs, monitor validation loss
- **Epochs**: Maximum 200 with early stopping
- **Callbacks**: ModelCheckpoint (save best model), ReduceLROnPlateau

### 5.3 GRU Model (Week 6)

#### Architecture Specifications
```python
Model: Sequential
├── GRU(units=64, return_sequences=True, input_shape=(lookback, n_features))
├── Dropout(0.2)
├── GRU(units=32, return_sequences=False)
├── Dropout(0.2)
├── Dense(32, activation='relu')
├── Dense(forecast_horizon * n_targets)
└── Reshape((forecast_horizon, n_targets))
```

#### Comparison Objective
- Evaluate GRU vs LSTM: training time, memory usage, prediction accuracy
- Hypothesis: GRU may achieve comparable performance with fewer parameters

### 5.4 Multi-Horizon Prediction Strategy

#### Approach: Direct Multi-Step Forecasting
- **Single model output**: Predict all horizons simultaneously
- **Output shape**: (batch_size, forecast_horizon, n_targets)
- **Forecast horizons**: t+1, t+3, t+6 hours
- **Alternative considered**: Recursive forecasting (use predictions as inputs) - higher error propagation risk

---

## 6. Evaluation Metrics

### 6.1 Primary Metrics

#### Root Mean Squared Error (RMSE)
```
RMSE = sqrt(mean((y_true - y_pred)²))
```
- **Unit**: Same as target variable (meters for SWH, seconds for DWP)
- **Target**: RMSE_SWH ≤ 0.3m (1h), ≤ 0.5m (6h)
- **Reported**: Per forecast horizon, per target variable

#### Mean Absolute Error (MAE)
```
MAE = mean(|y_true - y_pred|)
```
- **Purpose**: Robust to outliers, more interpretable
- **Reported**: Per forecast horizon, per target variable

### 6.2 Secondary Metrics

#### R² Score (Coefficient of Determination)
```
R² = 1 - (SS_res / SS_tot)
```
- **Purpose**: Measure proportion of variance explained
- **Target**: R² ≥ 0.85 for 1-hour predictions

#### Mean Absolute Percentage Error (MAPE)
```
MAPE = mean(|y_true - y_pred| / y_true) × 100%
```
- **Caution**: Undefined for y_true = 0, biased for small values

### 6.3 Surf-Specific Metrics

#### Good Surf Classification Accuracy
- **Definition**: Binary classification of "good surf" conditions
  - Good surf: SWH ≥ 1.2m AND DWP ≥ 12s
  - Poor surf: Otherwise
- **Metric**: Classification accuracy, precision, recall, F1-score
- **Purpose**: Evaluate practical utility for surfers

#### Extreme Wave Detection (Safety)
- **Definition**: Waves exceeding 3 meters (dangerous for most surfers)
- **Metric**: Precision and recall for extreme wave prediction
- **Purpose**: Safety-critical performance assessment

### 6.4 Comparative Metrics

#### Baseline Improvement
```
Improvement = ((RMSE_baseline - RMSE_model) / RMSE_baseline) × 100%
```
- **Requirement**: ≥20% improvement over persistence model
- **Reported**: For all models vs. best baseline

---

## 7. Implementation Requirements

### 7.1 Technology Stack

#### Core Dependencies
```python
python==3.9+
tensorflow==2.14+  # or tensorflow==2.15+
keras==2.14+
numpy==1.24+
pandas==2.0+
scikit-learn==1.3+
matplotlib==3.7+
seaborn==0.12+
```

#### Additional Libraries
```python
requests  # NOAA API access
beautifulsoup4  # Web scraping if needed
jupyter  # Notebook development
pytest  # Unit testing
```

### 7.2 Code Structure

```
surf-forecasting/
├── data/
│   ├── raw/               # Original NOAA downloads
│   ├── processed/         # Cleaned and feature-engineered data
│   └── splits/            # Train/val/test splits
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_models.ipynb
│   ├── 03_lstm_experiments.ipynb
│   └── 04_model_comparison.ipynb
├── src/
│   ├── data/
│   │   ├── download.py    # NOAA data fetching
│   │   ├── preprocessing.py
│   │   └── feature_engineering.py
│   ├── models/
│   │   ├── baseline.py
│   │   ├── lstm_model.py
│   │   ├── gru_model.py
│   │   └── train.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── visualization.py
│   └── utils/
│       ├── config.py
│       └── helpers.py
├── models/                # Saved model weights
├── results/               # Plots, metrics, analysis
├── tests/                 # Unit tests
├── requirements.txt
└── README.md
```

### 7.3 Configuration Management

#### config.yaml Example
```yaml
data:
  stations: ["46012", "46221"]
  start_date: "2018-01-01"
  end_date: "2024-12-31"
  features: ["WVHT", "DPD", "WSPD", "PRES"]
  target_variables: ["WVHT", "DPD"]
  
preprocessing:
  missing_threshold: 0.05
  outlier_method: "iqr"
  normalization: "standard"
  
model:
  lookback_hours: 24
  forecast_horizons: [1, 3, 6]
  lstm:
    layers: 2
    units: [64, 32]
    dropout: 0.2
  training:
    batch_size: 32
    epochs: 200
    learning_rate: 0.001
    patience: 20
```

### 7.4 Data Pipeline Requirements

#### Pipeline Stages
1. **Download**: Fetch NOAA data via API or bulk download
2. **Validation**: Check data quality, identify missing values
3. **Cleaning**: Handle missing data, remove outliers
4. **Feature Engineering**: Create derived features
5. **Normalization**: Apply StandardScaler
6. **Sequence Creation**: Generate input-output pairs with lookback window
7. **Split**: Temporal train/val/test split

#### Reproducibility Requirements
- Set random seeds: `numpy.random.seed(42)`, `tf.random.set_seed(42)`
- Version control all code and configurations
- Log all hyperparameters and training runs
- Save data preprocessing artifacts (scalers, feature names)

---

## 8. Experimental Plan

### 8.1 Week-by-Week Implementation

#### Week 1-2: Data Acquisition and Exploration
**Deliverables:**
- [ ] Download 5+ years of data from stations 46012 and 46221
- [ ] Create data quality report: missing data %, outliers, temporal coverage
- [ ] Generate exploratory visualizations:
  - Time series plots of SWH and DWP
  - Distribution histograms
  - Correlation matrix heatmap
  - Seasonal decomposition plots
- [ ] Document data characteristics and cleaning decisions

#### Week 3-4: Baseline Models and Data Pipeline
**Deliverables:**
- [ ] Implement persistence model baseline
- [ ] Implement linear regression baseline
- [ ] Implement feedforward neural network baseline
- [ ] Build reproducible data preprocessing pipeline
- [ ] Create train/val/test splits (60/20/20)
- [ ] Establish baseline performance metrics
- [ ] Generate baseline results table and visualizations

**Baseline Results Table Format:**
| Model | Horizon | RMSE_SWH | MAE_SWH | RMSE_DWP | MAE_DWP |
|-------|---------|----------|---------|----------|---------|
| Persistence | 1h | - | - | - | - |
| Linear | 1h | - | - | - | - |
| FNN | 1h | - | - | - | - |

#### Week 5: LSTM Implementation and Tuning
**Deliverables:**
- [ ] Implement LSTM architecture for 1-hour predictions
- [ ] Hyperparameter grid search:
  - Number of LSTM layers
  - Units per layer
  - Dropout rate
  - Lookback window size
- [ ] Training curves visualization (loss vs epoch)
- [ ] Best model selection based on validation performance
- [ ] LSTM results table comparing hyperparameter configurations

#### Week 6: GRU and Multi-Horizon Forecasting
**Deliverables:**
- [ ] Implement GRU architecture
- [ ] Compare GRU vs best LSTM: accuracy, training time, parameters
- [ ] Extend best-performing model to 3-hour and 6-hour horizons
- [ ] Multi-horizon performance analysis
- [ ] Model comparison table (LSTM vs GRU across all horizons)

#### Week 7: Advanced Analysis and Interpretation
**Deliverables:**
- [ ] Feature importance analysis (permutation importance or SHAP)
- [ ] Prediction vs actual scatter plots and time series plots
- [ ] Error distribution analysis:
  - Error vs wave height
  - Error vs forecast horizon
  - Temporal error patterns (seasonal, diurnal)
- [ ] Residual plots and autocorrelation analysis
- [ ] Good surf classification performance report
- [ ] Failure case analysis: when/why does the model fail?

#### Week 8: Deployment and Documentation
**Deliverables:**
- [ ] Final technical report with all results
- [ ] Model performance summary dashboard
- [ ] (Optional) Simple web interface for live predictions
- [ ] Code documentation and README
- [ ] Final presentation slides
- [ ] Recommendations for future work

### 8.2 Success Criteria by Week

| Week | Must Have | Nice to Have |
|------|-----------|--------------|
| 2 | Data downloaded, EDA complete | Data quality automation |
| 4 | Baseline models trained, pipeline working | Feature engineering experiments |
| 5 | LSTM beating baseline by 20%+ | Attention mechanisms |
| 6 | Multi-horizon predictions working | Ensemble methods |
| 7 | Complete analysis and visualizations | Interactive dashboards |
| 8 | Final report and presentation | Deployed demo app |

---

## 9. Risk Assessment and Mitigation

### 9.1 Technical Risks

| Risk | Likelihood | Impact | Mitigation Strategy |
|------|------------|--------|---------------------|
| Insufficient training data | Low | High | Use 5+ years of data; augmentation if needed |
| High missing data rate | Medium | High | Multiple station redundancy; interpolation methods |
| Overfitting on limited data | High | Medium | Regularization (dropout, L2); early stopping; cross-validation |
| Poor generalization to 6h horizon | Medium | High | Increase model capacity; add more features; ensemble |
| Computational constraints | Low | Medium | Use Google Colab Pro; optimize batch sizes |
| Class imbalance (rare good surf) | Medium | Low | Weighted loss; stratified sampling; focus on appropriate metrics |

### 9.2 Timeline Risks

| Risk | Mitigation |
|------|------------|
| Slow model training | Start with smaller models; use GPU acceleration |
| Debugging delays | Implement unit tests early; modular code design |
| Scope creep | Clearly define MVP; defer stretch goals to Week 8 |
| Final week crunch | Complete core deliverables by Week 7 |

---

## 10. Evaluation and Validation

### 10.1 Model Validation Strategy

#### K-Fold Time Series Cross-Validation (Optional)
- **Method**: Walk-forward validation with expanding window
- **Folds**: 5 temporal folds
- **Purpose**: Assess model stability across different time periods

#### Hold-out Test Set Evaluation
- **Primary method**: Final evaluation on unseen 2024 Q3-Q4 data
- **No hyperparameter tuning** on test set
- **Report confidence intervals** using bootstrap resampling

### 10.2 Statistical Significance Testing

#### Paired t-test for Model Comparison
- **Null hypothesis**: Model A and Model B have equal performance
- **Alternative**: Model A significantly outperforms Model B
- **Significance level**: α = 0.05
- **Apply to**: RMSE differences across test set predictions

### 10.3 Visualization Requirements

#### Required Plots
1. **Training curves**: Loss vs epoch (train and validation)
2. **Prediction plots**: Time series overlay of actual vs predicted
3. **Scatter plots**: Predicted vs actual (with R² and regression line)
4. **Error distributions**: Histogram and box plots of residuals
5. **Horizon comparison**: RMSE bar chart across 1h, 3h, 6h forecasts
6. **Feature importance**: Bar chart of top 10 features
7. **Confusion matrix**: For good surf classification

---

## 11. Deployment Considerations (Stretch Goal)

### 11.1 Inference Requirements
- **Latency**: < 1 second for single prediction
- **Input**: Most recent 24 hours of buoy data
- **Output**: JSON with forecasted SWH and DWP for 1h, 3h, 6h
- **Model format**: SavedModel (TensorFlow) or ONNX

### 11.2 Simple Web Interface (Optional)
- **Frontend**: HTML/CSS/JavaScript with Chart.js
- **Backend**: Flask or FastAPI
- **Features**:
  - Station selection dropdown
  - Real-time latest forecast display
  - Historical prediction accuracy charts
  - "Is it good surf?" indicator

---

## 12. Documentation Requirements

### 12.1 Code Documentation
- Docstrings for all functions (Google style)
- Type hints for function signatures
- Inline comments for complex logic
- README with setup instructions

### 12.2 Final Report Sections
1. **Abstract**: 250-word summary
2. **Introduction**: Motivation, problem statement, significance
3. **Related Work**: Literature review (3-5 papers)
4. **Methodology**: Data, features, architectures, training
5. **Experiments**: Hyperparameter search, ablation studies
6. **Results**: Quantitative metrics, visualizations
7. **Discussion**: Interpretation, limitations, future work
8. **Conclusion**: Key findings and contributions
9. **References**: Academic citations
10. **Appendix**: Additional plots, hyperparameter tables

### 12.3 Presentation Requirements
- 15-20 slides
- 10-minute presentation + 5-minute Q&A
- Key slides: Problem, Data, Architecture, Results, Demo

---

## 13. Future Work and Extensions

### 13.1 Immediate Extensions
- Additional buoy stations (expand to East Coast, Hawaii)
- Weather data integration (wind forecasts, swell models)
- Transformer-based architectures for longer sequences
- Ensemble methods combining multiple models

### 13.2 Long-term Vision
- Multi-location spatial modeling (CNN-LSTM)
- Real-time data ingestion pipeline
- Mobile app deployment
- Integration with surf forecasting APIs (Surfline, Magic Seaweed)
- Surfer-specific personalization (skill level, preferences)

---

## Appendix A: Performance Benchmarks

### Reference Performance from Literature
- **James et al. (2018)**: LSTM achieved RMSE = 0.25m for 3h wave height forecasts
- **Numerical models**: Typically RMSE = 0.3-0.5m for 6-24h forecasts
- **Persistence baseline**: RMSE increases by ~0.1m per forecast hour

### Target Performance Matrix
| Horizon | Target RMSE (SWH) | Target MAE (SWH) | Target R² |
|---------|-------------------|------------------|-----------|
| 1 hour  | ≤ 0.30m           | ≤ 0.20m          | ≥ 0.85    |
| 3 hours | ≤ 0.40m           | ≤ 0.30m          | ≥ 0.75    |
| 6 hours | ≤ 0.50m           | ≤ 0.40m          | ≥ 0.65    |

---

## Appendix B: NOAA Data Access Details

### API Endpoint Example
```
https://www.ndbc.noaa.gov/data/realtime2/{station_id}.txt
```

### Data Format
- **File type**: Space-delimited text
- **Header rows**: 2 (variable names and units)
- **Time format**: YYYY MM DD hh mm
- **Missing data**: 99.0 or 999.0 depending on variable

### Bulk Download
```bash
wget https://www.ndbc.noaa.gov/view_text_file.php?filename=46012h{year}.txt.gz&dir=data/historical/stdmet/
```

---