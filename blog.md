# What Makes a Footballer Worth €100 Million? We Built an AI to Find Out.

*Spoiler: it's not just the goals.*

---

Let's say you're a football club director. It's transfer window season. Your scouts are hyped about a 24-year-old winger from Sevilla — great feet, fast, good numbers — and his agent is asking for €45M.

Is that fair? Overpriced? A steal?

Right now, most clubs answer that question with spreadsheets, gut feel, and negotiation tactics. We decided to answer it with data.

Using the **FIFA 24 Player Stats Dataset** — 5,682 players, 40 attributes each, everything from sprint speed to composure — we built machine learning models to predict a player's market value in euros. Here's what we learned, what surprised us, and why our first three attempts completely failed.

---

## The Data: FIFA Knows Its Players

The dataset is surprisingly rich. Each player has physical stats (height, weight, age), technical skills (ball control, dribbling, finishing), mental attributes (reactions, composure), defensive skills, passing metrics, and goalkeeping ratings. Plus — crucially — their **actual market value** in euros.

5,682 players. 689 clubs. 135 countries. England has the most players (516), which tells you everything about the size of the English football pyramid.

Before touching a single model, we did what every good analyst should do: **look at the target variable**.

The distribution of player values is *aggressively* right-skewed. Most players sit in the €1–5M range. A handful of superstars balloon the tail toward €100M+. If you plot it, it looks less like a bell curve and more like a ski jump.

Why does this matter? Because the math of regression *assumes* your target looks like a bell. When it doesn't, your model quietly breaks — not loudly with errors, but subtly, with bad predictions and misleading statistics. This single insight became the most important decision in the entire project.

---

## Round 1 — Linear Regression: Honest, Simple, and Humbled

Every good experiment starts with a baseline. We threw all 36 features into a standard linear regression and asked it to predict player value.

**Result: R² = 0.39.**

To translate: the model explained just 39% of the variance in player values. For 36 predictors, that's underwhelming. The predictions were off by €3.2M on average.

But *why* did it fail? Two reasons, both visible in the data upfront:

**First** — multicollinearity. Football skill ratings are massively correlated with each other. `ball_control` and `dribbling` have a correlation of **0.945**. That's not "highly correlated" — that's "statistically almost the same thing." When you feed two near-identical columns into a linear model, it can't tell them apart and assigns arbitrary coefficients that cancel each other out. The math breaks.

**Second** — that ski-jump distribution. The model had no idea how to handle the log-normal shape of player values.

Linear regression was honest enough to tell us exactly where it was struggling. Time to fix both problems.

---

## Round 2 — OLS + Transformations: The Breakthrough Hidden in Plain Sight

The fix to the distribution problem turned out to be elegant: **take the log of player values before training**.

If values follow a log-normal distribution, then `log(value)` follows a normal distribution. One line of code, and suddenly the math works the way it's supposed to.

```python
log_model = sm.OLS(np.log(y), X).fit()
```

**Result: R² jumps to 0.83.**

That's not an incremental improvement — that's doubling the explanatory power *without changing a single feature*. Just fixing the mathematical assumption about the target's shape.

We also tried **Box-Cox transformation**, which finds the mathematically optimal power to apply to the target (log is just one special case). Box-Cox pushed R² to **0.84** — marginally better, with a more nuanced transformation.

The lesson here is one worth tattooing on every data scientist's desk: *understand your target distribution before you touch a model*.

But we'd hit a ceiling. Even with transformed targets, linear models are inherently straight lines. Football player value has curved, conditional, non-linear relationships that no amount of mathematical gymnastics would fix. `age` doesn't relate to value linearly — there's a prime-age window, a plateau, then a sharp decline. A linear coefficient can't capture that.

---

## Round 3 — Ridge & Lasso: Fighting Multicollinearity

Before abandoning linear models entirely, we tried one more approach: **regularization**.

Ridge and Lasso regression add a penalty term to the training process that shrinks coefficients toward zero — forcing the model to be more conservative about which features it trusts. This directly addresses the multicollinearity problem.

We used **cross-validation to find the optimal regularization strength** (instead of the default value, which is essentially a coin flip).

**Results:**
- Ridge RMSE: ~2.5–3.0 | R²: ~0.45
- Lasso RMSE: ~2.5–3.0 | R²: ~0.42

An improvement over raw linear regression, but still well below the transformed OLS. And here's what Lasso revealed when we checked which features it kept vs. zeroed out: `weight` dropped to exactly zero. Not important at all, once you control for everything else. Meanwhile, `reactions` and `composure` — mental attributes — survived as strong predictors.

That's an insight worth pausing on: **the market pays for mental sharpness, not just physical ability**.

Still, linear models had given us everything they had. Time to bring in the heavy machinery.

---

## Round 4 — Random Forest: Where the Game Changed

Random Forest trains hundreds of decision trees, each on a different random slice of the data, then averages their predictions. The result is a model that can learn *curves*, *thresholds*, and *interactions* that linear models fundamentally cannot.

And crucially — it handles multicollinearity naturally. Correlated features just take turns being used across different trees. The model doesn't care.

We tuned it properly using **randomized search over 25 hyperparameter combinations** with 5-fold cross-validation.

**Result: RMSE = 1.45 | R² = 0.875.**

The jump from R² = 0.45 (Ridge) to **R² = 0.875** (Random Forest) is staggering. The average prediction error dropped from €3M+ to under **€1.5M**. The residual plots looked dramatically cleaner.

Feature importance from the Random Forest confirmed what Lasso had hinted at: **`reactions` is the single most predictive feature for player value**. More than pace, more than finishing, more than physical attributes. The market, it turns out, prices *football intelligence* above almost everything else.

---

## Round 5 — XGBoost: Squeezing Out Every Last Drop

If Random Forest is "wisdom of the crowd," XGBoost is "iterative error correction." Each new tree in XGBoost specifically targets the mistakes of all previous trees — a process called gradient boosting. It also has built-in L1/L2 regularization, making it simultaneously powerful and controlled.

We ran **30 hyperparameter combinations** tuning learning rate, tree depth, subsampling, and regularization terms.

**Result: RMSE = 1.45 | R² = 0.876.**

Essentially identical to Random Forest — and that's actually a meaningful finding. When two very different model architectures converge to the same performance ceiling, it suggests you've hit the **natural predictability limit** of this feature set. To push further, you'd need richer data (contract years, club reputation, injury history), not a smarter algorithm.

---

## The Interpretability Question: SHAP to the Rescue

A €100M model is useless if no one trusts it. So we applied **SHAP values** — a mathematically grounded method to explain each individual prediction.

For every player, SHAP shows exactly which attributes pushed the predicted value up, and which pulled it down. The top drivers confirmed our intuition:

1. **Reactions** — the biggest positive driver
2. **Age** — young players get a premium; players past 30 see sharp value drops
3. **Composure** — clubs pay for players who don't crack under pressure
4. **Ball control & dribbling** — technical mastery still commands a premium
5. **Height/Weight** — surprisingly *negative* predictors when everything else is controlled for

---

## What We Actually Learned

| Model | RMSE (€M) | R² |
|---|---|---|
| Linear Regression | 3.23 | 0.39 |
| OLS + Log Transform | ~1.9 | 0.83 |
| Ridge / Lasso | ~2.7 | ~0.44 |
| **Random Forest** | **1.45** | **0.875** 🏆 |
| **XGBoost** | **1.45** | **0.876** 🏆 |
| XGBoost (Hypothesis Features) | 1.63 | 0.844 |

Three takeaways that generalize far beyond this project:

**1. Your target distribution matters more than your model choice.** Applying a log transform doubled R² with zero new features. Understanding your data is more valuable than any algorithm.

**2. Multicollinearity kills linear models.** `ball_control` and `dribbling` at r=0.945 means linear regression is effectively broken before it starts. Tree models don't have this problem.

**3. Intelligence > Athleticism (at least in the market).** `Reactions` and `composure` outperform `sprint_speed` and `strength` as predictors of market value. Clubs know what they're paying for.

---

Football analytics has come a long way since Moneyball. We're not just counting shots on goal anymore — we're modeling how 40 attributes interact to produce a number that a club is willing to put on a transfer document.

Our model predicts that market value with an average error of €1.45M. Is it perfect? No. Does it know that a player's social media following also drives transfer fees? No.

But it's a start. And honestly, for a problem this complex — it's a pretty good one.

---

*Built with Python, scikit-learn, XGBoost, statsmodels, and a lot of Q-Q plots. Dataset: FIFA 24 Player Stats (5,682 players).*
