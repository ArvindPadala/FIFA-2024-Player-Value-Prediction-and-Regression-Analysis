# FIFA 2024 — Player Value Prediction & Regression Analysis

> **Predicting the market value of professional soccer players using FIFA 24 stats, advanced regression models, and ML techniques.**

---

## 📌 Project Overview

This project builds a machine learning pipeline to predict a player's **market value (in €M)** from in-game statistics sourced from the FIFA 24 Player Stats Dataset. It combines rigorous statistical analysis with state-of-the-art regression and ensemble techniques, making it useful for soccer clubs, analysts, and scouts to understand what drives player valuation.

---

## 📁 Repository Structure

```
FIFA-2024-Player-Value-Prediction-and-Regression-Analysis/
│
├── FIFA24_Enhanced.ipynb                          ← ✅ Enhanced notebook (use this)
├── FIFA24_Regression_Analysis_final.ipynb         ← Original analysis
│
├── player_stats.csv                               ← Dataset (required, not in repo)
│
├── FIFA24 - Regression Analysis and Player Value Prediction.pptx
├── Regression Project report.docx
├── hypothesis test questions.docx
│
├── requirements.txt                               ← Python dependencies
├── README.md
└── .gitignore
```

---

## 📊 Dataset

| Property | Value |
|---|---|
| **Source** | FIFA 24 Player Stats |
| **Players** | 5,682 |
| **Features** | 40 (physical, technical, mental, goalkeeping, identity) |
| **Target** | `value(Million)` — market value in €M |
| **Nationalities** | 135 |
| **Clubs** | 689 |

**Key feature categories:**

| Category | Features |
|---|---|
| Physical | `height`, `weight`, `strength`, `stamina`, `jumping` |
| Pace | `acceleration`, `sprint_speed` |
| Attacking | `finishing`, `shot_power`, `long_shots`, `volleys`, `penalties` |
| Passing | `short_pass`, `long_pass`, `crossing`, `vision` |
| Dribbling | `ball_control`, `dribbling`, `agility`, `balance` |
| Defending | `stand_tackle`, `slide_tackle`, `interceptions`, `aggression` |
| Goalkeeping | `gk_positioning`, `gk_diving`, `gk_handling`, `gk_kicking`, `gk_reflexes` |
| Mental | `reactions`, `composure` |

---

## 🔬 Methodology

### 1. Data Cleaning
- Removed `marking` column (stored as object type — incompatible with numeric models)
- Fixed target variable `value` — parsed currency strings (e.g., `$1.000.000`) into float values and scaled to millions
- Removed 64 statistical outliers using Z-score threshold of 4

### 2. Exploratory Data Analysis
- Target distribution shows **heavy right-skew** → log-normal distribution → justifies transformations
- **VIF analysis** reveals extreme multicollinearity (e.g., `ball_control` ↔ `dribbling`: r = 0.945)
- Player **position classification** (GK vs Outfield) based on GK skill ratios
- Top/Bottom 10 player value visualizations

### 3. Feature Engineering
Five composite features derived from raw attributes:

| Feature | Formula |
|---|---|
| `pace` | (acceleration + sprint_speed) / 2 |
| `attacking` | (finishing + shot_power + long_shots + volleys + penalties) / 5 |
| `defending` | (stand_tackle + slide_tackle + interceptions) / 3 |
| `passing` | (short_pass + long_pass + crossing) / 3 |
| `physical` | (strength + stamina + jumping) / 3 |

### 4. Models Trained

| Model | Strategy |
|---|---|
| Linear Regression | Baseline |
| OLS (with constant) | Statsmodels — proper centered R² |
| OLS + Log Transform | Log(y) target — inverse-transformed for RMSE |
| OLS + Box-Cox | Optimal lambda stored for inverse-transform |
| Ridge (Tuned) | `RidgeCV` — optimal alpha from 50 log-spaced candidates |
| Lasso (Tuned) | `LassoCV` — L1 regularization + automatic feature selection |
| Random Forest (Tuned) | `RandomizedSearchCV` over 25 hyperparameter combos |
| XGBoost (Tuned) | `RandomizedSearchCV` over 30 hyperparameter combos |
| XGBoost + Ridge Features | 14 features selected via `SelectFromModel` |
| XGBoost + PCA | 10 principal components |
| XGBoost + Hypothesis Feats | 21 domain-significant features |

### 5. Evaluation
- All models evaluated on held-out 30% test set
- **5-fold cross-validation** RMSE reported for all final models
- Metrics: RMSE (€M), R², Explained Variance Score
- Q-Q residual plots for normality assessment

### 6. Hypothesis Testing
Hypothesis testing validates the significance of player roles (position-dependent dynamics) on predicted market values.

### 7. SHAP Interpretability
TreeExplainer SHAP values explain individual predictions from the best XGBoost model — both global importance ranking and directional beeswarm plot.

---

## 📈 Results Summary

| Model | RMSE (€M) | R² |
|---|---|---|
| Linear Regression | 3.23 | 0.39 |
| OLS (with constant) | ~3.22 | ~0.40 |
| OLS + Log Transform | ~1.8* | ~0.83 |
| OLS + Box-Cox | ~1.9* | ~0.84 |
| Ridge (Tuned) | ~2.5–3.0 | ~0.45+ |
| Lasso (Tuned) | ~2.5–3.0 | ~0.42+ |
| **Random Forest (Tuned)** | **≤1.45** | **≥0.875** 🏆 |
| **XGBoost (Tuned)** | **≤1.45** | **≥0.875** 🏆 |
| XGBoost + Hyp. Features | ~1.63 | ~0.844 |

*\*Inverse-transformed from log/Box-Cox scale for fair RMSE comparison.*

**Key finding:** Ensemble methods (RF, XGBoost) outperform linear models by >2× in both RMSE and R² due to their ability to model non-linear relationships and tolerance for multicollinearity.

---

## 🚀 Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Run the Enhanced Notebook
```bash
jupyter notebook FIFA24_Enhanced.ipynb
```
Then: **Kernel → Restart & Run All**

> ⚠️ Make sure `player_stats.csv` is in the project root directory before running.

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| `pandas` | Data loading, wrangling, feature engineering |
| `numpy` | Numerical operations |
| `matplotlib` / `seaborn` | Visualization |
| `statsmodels` | OLS, VIF, statistical summaries |
| `scipy` | Z-scores, Box-Cox transform, hypothesis tests |
| `scikit-learn` | Regression models, CV, feature selection, PCA |
| `xgboost` | Gradient boosted trees |
| `shap` | Model interpretability |

---

## 💡 Key Insights

1. **Player value follows a log-normal distribution** — log/Box-Cox transforms dramatically improve OLS model fit
2. **Multicollinearity is severe** — skill ratings are highly inter-correlated (e.g., ball_control ↔ dribbling r=0.945); linear models suffer significantly
3. **Ensemble methods dominate** — Random Forest and XGBoost handle multicollinearity natively via tree structures
4. **Reactions & composure are the strongest linear predictors** — mental attributes matter more than raw physical stats for value
5. **GK vs Outfield split matters** — goalkeepers command different salary structures than outfield players
6. **Hyperparameter tuning yields measurable gains** over default configurations

---

## 📌 Future Work

- **Position-specific models**: Train separate pipelines for GK, Defenders, Midfielders, Forwards
- **Temporal modelling**: Incorporate FIFA 22/23 data to model career trajectory → value trend
- **External features**: League prestige, club reputation, injuries, contract years remaining
- **Deep learning**: Feedforward neural network on tabular data
- **Model stacking**: Blend RF + XGBoost + OLS with a meta-learner

---

## 📄 License

This project is for academic and educational purposes using publicly available FIFA 24 player statistics data.
