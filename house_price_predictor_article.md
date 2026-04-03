# Building a Production-Ready House Price Predictor with scikit-learn

> *From raw data to deployment — a step-by-step guide to building a clean, correct, and reliable ML pipeline*

---

## Introduction: Why Most ML Code Breaks in Production

You've probably seen it before: a notebook that scores an impressive R² of 0.90 in training, but collapses the moment it meets real data. The model was perfectly fine — the *code around it* was broken.

Most ML tutorials focus on getting good numbers. Few focus on getting *correct* numbers. There's a critical difference, and it comes down to one deceptively simple rule: **never let your test set touch your training logic.**

In this article, we'll walk through building an end-to-end house price prediction system on the California Housing Dataset — the right way. We'll use scikit-learn Pipelines not just as a convenience, but as a guarantee of correctness. We'll hunt down data leakage, build custom transformers, tune hyperparameters systematically, and validate inputs before every prediction.

By the end, you'll have a model you can actually trust — and the mental framework to build more of them.

---

## The Dataset: California Housing, 1990

The [California Housing Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html), included in scikit-learn, contains 20,640 census block records from the 1990 US Census. Each row represents a neighborhood, and the target variable (`MedHouseVal`) is the median house value in that block, expressed in units of $100K.

The eight features capture the socioeconomic and physical character of each block:

| Feature | Description |
|---|---|
| `MedInc` | Median household income |
| `HouseAge` | Median house age |
| `AveRooms` | Average number of rooms per household |
| `AveBedrms` | Average number of bedrooms per household |
| `Population` | Block population |
| `AveOccup` | Average household occupancy |
| `Latitude` | Geographic latitude |
| `Longitude` | Geographic longitude |

After loading:

```
Shape: (20640, 9)
Target range: $14,999 – $500,001
Missing values: 0
```

---

## Step 1: Split First, Transform Never (the Golden Rule)

This is where most tutorials go wrong. Before touching a single value, we split the data:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

**Why does this matter?** Every transformation that *learns something from data* — scaling, outlier bounds, imputation statistics — must be fitted only on the training set. If you scale your entire dataset before splitting, the scaler has already "seen" the test set, and your model will appear to perform better than it truly does.

This is called **data leakage**, and it's the single most common source of inflated ML metrics in practice.

> ✅ **Golden Rule:** `train_test_split` is always the first operation after loading raw data.

*[VISUAL SUGGESTION: A simple diagram showing the correct order: Load Data → Split → Fit on Train → Transform Train+Test → Model. Caption: "The correct ML workflow — split first, everything else second."]*

---

## Step 2: Outlier Capping as a Proper Transformer

The California Housing Dataset has a well-known quirk: house values are hard-capped at $500K, creating an artificial spike at the top of the distribution. Naively clipping outliers before splitting introduces leakage — your clipping bounds are computed using future (test) data.

The fix: wrap the outlier logic in a proper scikit-learn transformer.

```python
class IQRCapper(BaseEstimator, TransformerMixin):
    """
    Clips each feature to [Q1 - 1.5*IQR, Q3 + 1.5*IQR] bounds.
    Bounds are fitted only on training data to prevent leakage.
    """
    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        q1 = X.quantile(0.25)
        q3 = X.quantile(0.75)
        iqr = q3 - q1
        self.lower_ = q1 - 1.5 * iqr
        self.upper_ = q3 + 1.5 * iqr
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        return np.clip(X, self.lower_.values, self.upper_.values)
```

**What changes?** Now the IQR bounds are computed during `fit()` — which only sees training data. The same bounds are then applied to the test set during `transform()`, with no peeking allowed.

*[VISUAL SUGGESTION: Side-by-side histograms of the price distribution before and after IQR capping. Title: "Price Distribution: Before vs. After Outlier Capping". Generated with matplotlib using `plt.hist()`.]*

---

## Step 3: Feature Engineering — Teaching the Model What Matters

Raw features often tell an incomplete story. The model sees numbers; we need to encode *meaning*.

```python
class HousingFeatureEngineer(BaseEstimator, TransformerMixin):
    def transform(self, X):
        X = pd.DataFrame(X, columns=[...])
        X['rooms_per_bedroom'] = X['AveRooms'] / X['AveBedrms'].replace(0, np.nan)
        X['pop_density_proxy'] = X['Population'] / X['AveOccup'].replace(0, np.nan)
        X['dist_sf']           = np.sqrt((X['Latitude'] - 37.77)**2 + (X['Longitude'] + 122.42)**2)
        X['income_sq']         = X['MedInc'] ** 2
        return X.values
```

Each new feature encodes domain knowledge:

- **`rooms_per_bedroom`** — A ratio of 10 tells a very different story than `AveRooms = 10` alone. It signals spaciousness.
- **`pop_density_proxy`** — Neighborhood density correlates with infrastructure, pricing pressure, and lifestyle.
- **`dist_sf`** — Euclidean distance to San Francisco. Proximity to a major economic hub drives prices non-linearly.
- **`income_sq`** — Squares the income signal to capture the disproportionate price premium in wealthy neighborhoods.

By placing this logic inside a transformer, it runs automatically during cross-validation, retraining, and prediction — you can never accidentally forget a step.

---

## Step 4: The Full Pipeline — One Object to Rule Them All

All five steps are chained into a single scikit-learn Pipeline:

```
raw data
  → IQRCapper        (clip outliers using TRAIN bounds)
  → FeatureEngineer  (add 4 new features)
  → SimpleImputer    (fill any NaN from zero-division guards)
  → StandardScaler   (normalize to mean=0, std=1)
  → GradientBoosting (predict)
```

```python
pipeline = Pipeline([
    ('capper',    IQRCapper(factor=1.5)),
    ('engineer',  HousingFeatureEngineer()),
    ('imputer',   SimpleImputer(strategy='median')),
    ('scaler',    StandardScaler()),
    ('regressor', GradientBoostingRegressor(...)),
])
```

*[VISUAL SUGGESTION: A flow diagram of the pipeline steps, left to right. Each step is a labeled box with an arrow. Title: "The Full sklearn Pipeline". Tools: matplotlib or a simple SVG.]*

**Why Gradient Boosting?** House prices are driven by complex, non-linear interactions — income and location interact, age and rooms interact. Gradient Boosting builds trees sequentially, each correcting the errors of the previous one. It's the workhorse of tabular ML competitions for good reason.

---

## Step 5: Systematic Hyperparameter Tuning

Hard-coding `n_estimators=300` is not a strategy — it's a guess. A better approach is `RandomizedSearchCV`, which samples random combinations from a parameter grid and evaluates each using cross-validation:

```python
param_dist = {
    'regressor__n_estimators':  [100, 200, 300, 400, 500],
    'regressor__max_depth':     [3, 4, 5, 6, 7],
    'regressor__learning_rate': [0.03, 0.05, 0.08, 0.1, 0.15],
    'regressor__subsample':     [0.7, 0.8, 0.9, 1.0],
    'capper__factor':           [1.5, 2.0, 2.5],  # tuning our custom step too!
}
```

Note that we're also tuning `capper__factor` — the IQR multiplier. This is only possible because the capper is inside the Pipeline.

After 20 iterations (60 total fits), the best configuration found:

```
Best CV R²: 0.8407
Best params:
  regressor__subsample: 0.7
  regressor__n_estimators: 400
  regressor__max_depth: 6
  regressor__learning_rate: 0.08
  capper__factor: 2.5
```

---

## Step 6: Final Evaluation — Touch the Test Set Once

The test set has been untouched throughout. We use it exactly once, at the very end:

```
5-Fold CV R²: 0.8436 ± 0.0054

Test Metrics:
  RMSE: 0.4476   ($44,757)
  MAE:  0.2925   ($29,249)
  R²:   0.8471
```

The model explains **84.7%** of variance in house prices, with a mean absolute error of ~$29K. More importantly, the CV score and test score are tightly aligned — there's no suspicious gap that would suggest leakage or overfitting.

*[VISUAL SUGGESTION: Scatter plot of Actual vs. Predicted prices, with a red dashed diagonal line representing perfect fit. Caption: "Actual vs. Predicted House Prices (R² = 0.847)". Use matplotlib `plt.scatter()`.]*

---

## Step 7: Feature Importance — Two Methods, One Right Answer

**Method 1 — Built-in `feature_importances_`** measures how much each feature reduced impurity across all trees. It's fast, but has a known bias: it systematically overestimates features with many unique values.

**Method 2 — Permutation Importance** randomly shuffles one feature at a time and measures how much the model's R² drops. This directly measures prediction impact — no internal tree mechanics involved. It's the preferred method for reporting.

Top 5 features by permutation importance:

| Feature | Mean R² Drop | Interpretation |
|---|---|---|
| `Latitude` | 2.23 | Location is king |
| `Longitude` | 1.08 | Geographic position |
| `dist_sf` | 0.80 | Proximity to SF (engineered!) |
| `income_sq` | 0.24 | Non-linear income effect (engineered!) |
| `AveOccup` | 0.13 | Household density |

Two of the top five features are engineered — a direct payoff from Step 3.

*[VISUAL SUGGESTION: Side-by-side horizontal bar charts — left shows built-in importances, right shows permutation importances with error bars. Dark background theme. Title: "Feature Importance: Built-in vs. Permutation". Use matplotlib with `barh()`.]*

---

## Step 8: Input Validation Before Every Prediction

Production systems fail silently. A missing field doesn't raise an error — it propagates as `NaN`, produces a prediction, and nobody knows the output is meaningless.

The solution is explicit validation before calling `.predict()`:

```python
def validate_input(data: dict) -> pd.DataFrame:
    REQUIRED_FIELDS = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
                       'Population', 'AveOccup', 'Latitude', 'Longitude']

    missing = [f for f in REQUIRED_FIELDS if f not in data]
    if missing:
        raise ValueError(f'Missing required fields: {missing}')

    if data['MedInc'] < 0:
        raise ValueError(f'MedInc cannot be negative: {data["MedInc"]}')

    # ... additional domain checks

    return pd.DataFrame([{k: data[k] for k in REQUIRED_FIELDS}])
```

Demo predictions with the saved model:

```
🏷  House near SF    → Predicted price: $385,778
🏷  Inland house     → Predicted price: $79,774
✅  Missing 'Latitude' → Caught error: Missing required fields: ['Latitude']
```

---

## Key Takeaways

**1. Split before everything else.** Data leakage is silent and optimistic. Any transformation that learns from data must happen inside a Pipeline, after the split.

**2. Custom transformers are first-class citizens.** Wrapping feature engineering and outlier handling in `BaseEstimator` + `TransformerMixin` makes them reusable, tunable, and pipeline-compatible.

**3. Pipelines are correctness guarantees, not just convenience.** They ensure that every cross-validation fold respects the train/test boundary — automatically.

**4. Hyperparameters should be searched, not guessed.** `RandomizedSearchCV` is fast enough for most problems and far more principled than manual tuning.

**5. Use permutation importance for interpretability.** Built-in `feature_importances_` has biases. Permutation importance measures what actually matters for predictions.

**6. Validate inputs before inference.** Silent failures are worse than loud ones. Know what your model expects and enforce it explicitly.

---

## What's Next?

This notebook demonstrates solid ML engineering fundamentals. A few natural next steps:

- **Time-based splitting** — If data has a temporal component, use `TimeSeriesSplit` instead of random splits
- **SHAP values** — For per-prediction explainability beyond global feature importance
- **Model serving** — Wrap the pipeline in a FastAPI endpoint and deploy it
- **Data drift monitoring** — Track whether the input distribution shifts over time in production

The full notebook (with all outputs) is available on GitHub.

---

*This article is based on a working Colab notebook. All metrics were computed on held-out test data using the methods described above.*
