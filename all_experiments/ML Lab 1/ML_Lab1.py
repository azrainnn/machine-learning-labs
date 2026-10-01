# %% [markdown]
# # Lab 1 of 8 — Feature engineering and feature selection
#
# *Check the tools, understand the data, build features, and choose among them.*
#
# **Goal.** Check the environment, engineer features, compare scaling, and select features without leakage.
#
# The notebook finds `all_datasets/` by searching its own folder and the folders above it.

# %% [markdown]
# ## Setup

# %%
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, pairwise_distances, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Extras used beyond the lab's own setup (naive baseline in Part D, cross-validation in Try it yourself):
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import StratifiedKFold, cross_val_score


def find_data_dir(name="all_datasets", marker="titanic_dataset.csv"):
    """Search the script/working folder and their parents for a data folder holding `marker`."""
    starts = [Path.cwd()]
    try:
        starts.insert(0, Path(__file__).resolve().parent)  # not defined inside a notebook
    except NameError:
        pass
    for start in starts:
        for folder in [start, *start.parents]:
            if (folder / name / marker).is_file():
                return folder / name
    searched = ", ".join(str(s) for s in starts)
    raise FileNotFoundError(f"Could not find {name}/{marker} in or above: {searched}")


# The lab's original `Path.cwd().parent` only works for one folder depth (notebooks directly in
# all_experiments/); searching upward works for any depth.
DATA = find_data_dir()
PROJECT_ROOT = DATA.parent
print("Using data from:", DATA)

# %% [markdown]
# ## Part A — Environment check and a first look at the data

# %% [markdown]
# Record package versions for reproducibility.

# %%
print("NumPy:", np.__version__)
print("pandas:", pd.__version__)
print("Matplotlib:", matplotlib.__version__)
print("scikit-learn:", sklearn.__version__)

# %% [markdown]
# Titanic has one row per passenger; `Survived=1` denotes survival. Inspect shape, types, missingness, and class balance.

# %%
df = pd.read_csv(DATA / "titanic_dataset.csv")
print("shape:", df.shape)
print(df.head())
print("\nData types:\n", df.dtypes)
print("\nSurvival rate:\n", df["Survived"].value_counts(normalize=True))

# %% [markdown]
# Plot missing-value percentages by column.

# %%
missing = df.isna().mean().sort_values(ascending=False)
missing = missing[missing > 0]
plt.figure(figsize=(5, 3))
plt.barh(missing.index, missing.values * 100)
plt.xlabel("Share of rows missing (%)")
plt.title("Titanic: where the gaps are")
plt.tight_layout()
plt.show()

# %% [markdown]
# **Inspect.** 891 passengers × 12 columns, and 38.4% survived, so the classes are moderately imbalanced (always predicting "died" would already be right 61.6% of the time). Three columns have gaps: `Cabin` (77.1% missing), `Age` (19.9%) and `Embarked` (0.2%, two rows). That is why imputation has to be part of the modelling pipeline, and why `Cabin` is too sparse to use naively (it returns in *Try it yourself*).

# %% [markdown]
# ## Part B — Feature engineering

# %% [markdown]
# Derive family size, isolation, title, and log fare from prediction-time information. Each feature must exist *before* the outcome.

# %%
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.")[0].str.strip()
df["Title"] = df["Title"].where(df["Title"].isin(["Mr", "Mrs", "Miss", "Master"]), "Other")
df["LogFare"] = np.log1p(df["Fare"])
print(df[["Name", "Title", "FamilySize", "IsAlone", "Fare", "LogFare"]].head())
print("\nTitle counts:\n", df["Title"].value_counts())

# %% [markdown]
# Compare survival rates across engineered features.

# %%
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
df.groupby("Title")["Survived"].mean().sort_values().plot.barh(ax=axes[0])
axes[0].set_xlabel("Survival rate")
axes[0].set_title("Title carries the sex-and-age story in one column")
df.groupby("FamilySize")["Survived"].mean().plot.bar(ax=axes[1], rot=0)
axes[1].set_ylabel("Survival rate")
axes[1].set_title("Small families did best; large ones did worst")
plt.tight_layout()
plt.show()

# %% [markdown]
# **Inspect.** Survival climbs from 15.7% for `Mr` to 79.2% for `Mrs`, with `Miss` (69.8%) and `Master` (57.5%) in between: the title packs sex and rough age into one column. Passengers travelling alone survived at 30.4%, family sizes 2–4 did much better (55–72%), and sizes 5+ fell to 0–33%. Those largest families are tiny groups (6–22 passengers per size), so treat the right-hand bars as suggestive rather than precise.

# %% [markdown]
# Compare original and engineered features using the same model and split. Fit imputation, scaling, and encoding **inside** the training pipeline.

# %%
def make_preprocess(num_cols, cat_cols):
    return ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), num_cols),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), cat_cols),
    ])


y = df["Survived"]
feature_sets = {
    "raw columns": (["Age", "SibSp", "Parch", "Fare"], ["Pclass", "Sex", "Embarked"]),
    "raw + engineered": (["Age", "SibSp", "Parch", "LogFare", "FamilySize", "IsAlone"],
                         ["Pclass", "Sex", "Embarked", "Title"]),
}
for name, (num_cols, cat_cols) in feature_sets.items():
    X = df[num_cols + cat_cols]
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    model = Pipeline([
        ("preprocess", make_preprocess(num_cols, cat_cols)),
        ("model", LogisticRegression(max_iter=2000)),
    ]).fit(X_train, y_train)
    auc = roc_auc_score(y_valid, model.predict_proba(X_valid)[:, 1])
    print(f"{name:18s} validation ROC AUC = {auc:.3f}")

# %% [markdown]
# **Inspect.** Same estimator (logistic regression), same stratified 75/25 split, same seed: validation ROC AUC rises from 0.842 to 0.872. The gain measures the features (`Title`, `FamilySize`, `IsAlone`, `LogFare`), not a different model, and all of them are knowable at boarding, so none leaks the outcome. One split of 223 passengers is a noisy estimate; the cross-validated table in *Try it yourself* points the same way (0.851 → 0.873).

# %% [markdown]
# ## Part C — Scaling changes the geometry

# %% [markdown]
# Breast-cancer measurements have unequal scales: area reaches thousands while smoothness is below one. Compare their effect on distances.

# %%
cancer = pd.read_csv(DATA / "breast_cancer_dataset.csv")
feature_cols = [c for c in cancer.columns if c not in {"id", "diagnosis", "Unnamed: 32"}]
X_cancer = cancer[feature_cols]
three = ["area_mean", "texture_mean", "smoothness_mean"]
print(X_cancer[three].describe().loc[["mean", "std"]].round(3))

# %% [markdown]
# Compare raw and standardized distances among the first five tumors; darker heatmap cells indicate closer pairs.

# %%
raw = pairwise_distances(X_cancer.iloc[:5], metric="euclidean")
X_scaled = StandardScaler().fit_transform(X_cancer)
scaled = pairwise_distances(X_scaled[:5], metric="euclidean")
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
for ax, matrix, title in [(axes[0], raw, "Raw units"), (axes[1], scaled, "Standardized")]:
    im = ax.imshow(matrix, cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("tumour")
    ax.set_ylabel("tumour")
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{matrix[i, j]:.0f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8)
plt.suptitle("Same five tumours, two geometries")
plt.tight_layout()
plt.show()

# %% [markdown]
# Read the two closest pairs off each matrix directly (tumours are numbered as on the plot axes, starting at 0):

# %%
def nearest_pairs(matrix, n=2):
    """Return the n closest off-diagonal pairs as (distance, row, col)."""
    rows, cols = np.triu_indices_from(matrix, k=1)
    order = np.argsort(matrix[rows, cols])[:n]
    return [(float(matrix[rows[k], cols[k]]), int(rows[k]), int(cols[k])) for k in order]


for label, matrix in [("raw units", raw), ("standardized", scaled)]:
    (d1, i1, j1), (d2, i2, j2) = nearest_pairs(matrix)
    print(f"{label:12s}: closest = tumours {i1},{j1} (distance {d1:.2f}); "
          f"runner-up = tumours {i2},{j2} (distance {d2:.2f})")

# %% [markdown]
# **Inspect.** The three columns live on very different scales: `area_mean` averages 654.9 (std 351.9), `texture_mean` 19.3 (std 4.3) and `smoothness_mean` 0.096 (std 0.014). The raw distances use all 30 columns, and there the two area columns (`area_worst` and `area_mean`) supply nearly all of the squared distance among these five tumours, while `smoothness_mean` contributes essentially nothing.
#
# The closest pair changes from tumours 3/5 to 2/5 after scaling. The lab counts tumours from 1, while the plot axes and the code above count from 0, so `(2, 4)` → `(1, 4)` is the same statement. The raw result is clear-cut (164.15 against a runner-up near 277). The standardized result is a narrow call: `(1, 4)` at 4.38 edges out `(2, 4)` at 4.46 (about 2%), and because the heatmap labels are rounded to whole numbers both cells display as "4". The printed distances are what separate them. Units can change neighbours, which matters for any distance-based method such as k-NN or k-means.
#
# One caveat: the scaler here was fitted on all 569 tumours. That is fine for illustrating geometry, but it is exactly what you must **not** do before scoring a model (see question 1 below).

# %% [markdown]
# ## Part D — Feature selection without leakage

# %% [markdown]
# Predict `G3` for 649 students **at enrolment**. Exclude later grades `G1` and `G2`, leaving 30 candidate columns.

# %%
students = pd.read_csv(DATA / "student_performance_dataset.csv")
print("shape:", students.shape, " missing:", students.isna().sum().sum(),
      " duplicates:", students.duplicated().sum())
target = "G3"
leaky = ["G1", "G2"]
X_s = students.drop(columns=[target] + leaky)
y_s = students[target]
s_num = X_s.select_dtypes(include="number").columns.tolist()
s_cat = X_s.select_dtypes(exclude="number").columns.tolist()
print(len(s_num), "numeric and", len(s_cat), "text features available at enrolment")
Xs_train, Xs_valid, ys_train, ys_valid = train_test_split(X_s, y_s, test_size=0.25, random_state=42)

# %% [markdown]
# Compare ridge using all features or ten `SelectKBest` features, and lasso's sparse fit. Keep selection **inside** training pipelines.

# %%
def evaluate(pipeline, X_tr, X_va, label):
    pipeline.fit(X_tr, ys_train)
    mae = mean_absolute_error(ys_valid, pipeline.predict(X_va))
    print(f"{label:32s} validation MAE = {mae:.3f} grade points")
    return mae


results = {}
results["all 30 features"] = evaluate(Pipeline([
    ("preprocess", make_preprocess(s_num, s_cat)),
    ("model", Ridge(alpha=1.0)),
]), Xs_train, Xs_valid, "all 30 features")
results["SelectKBest, k=10"] = evaluate(Pipeline([
    ("preprocess", make_preprocess(s_num, s_cat)),
    ("select", SelectKBest(f_regression, k=10)),
    ("model", Ridge(alpha=1.0)),
]), Xs_train, Xs_valid, "SelectKBest, k=10")
lasso = Pipeline([
    ("preprocess", make_preprocess(s_num, s_cat)),
    ("model", Lasso(alpha=0.1)),
])
results["lasso (selects itself)"] = evaluate(lasso, Xs_train, Xs_valid, "lasso (selects itself)")
# Reference point (not plotted): predict the training-set mean grade for every student.
baseline = evaluate(DummyRegressor(), Xs_train, Xs_valid, "baseline: predict train mean")

# %% [markdown]
# A fourth run deliberately adds `G2` to expose leakage.

# %%
leak_num = s_num + ["G2"]
results["all 30 + G2 (leakage)"] = evaluate(Pipeline([
    ("preprocess", make_preprocess(leak_num, s_cat)),
    ("model", Ridge(alpha=1.0)),
]), Xs_train.assign(G2=students.loc[Xs_train.index, "G2"]),
    Xs_valid.assign(G2=students.loc[Xs_valid.index, "G2"]), "all 30 + G2 (leakage)")

# %%
coefs = pd.Series(lasso.named_steps["model"].coef_,
                  index=lasso.named_steps["preprocess"].get_feature_names_out())
# Lasso leaves ~1e-15 float dust on mirrored one-hot columns; a tolerance counts real selections only.
kept = coefs[coefs.abs() > 1e-8].sort_values()
print(f"lasso kept {len(kept)} of {len(coefs)} encoded columns")
print(kept.round(2).to_string())
# The lab's title claimed the leak "halves" the error; compute the measured drop instead.
leak_drop = 1 - results["all 30 + G2 (leakage)"] / results["all 30 features"]
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
axes[0].barh(list(results), list(results.values()))
axes[0].set_xlabel("Validation MAE (grade points)")
axes[0].set_title(f"Fewer features, same error; leaked G2, {leak_drop:.0%} less error")
axes[1].barh(kept.index, kept.values)
axes[1].axvline(0, color="grey", linewidth=0.8)
axes[1].set_title("What the lasso kept")
plt.tight_layout()
plt.show()

# %% [markdown]
# **Inspect.** With only enrolment-time information, ridge on all 30 columns (56 after one-hot encoding) has validation MAE 2.18 grade points. `SelectKBest` with k=10 gets 2.14 and the lasso 2.08, so similar or slightly better error from fewer columns (the lasso kept 16 of 56). On this one split that looks like a small win for selection; the 30-split check in *Try it yourself* shows it is not a reliable one, so read it as "no accuracy lost". For scale, predicting every student's grade as the training-set mean gives 2.37, so what is knowable at enrolment explains only a modest share of `G3`.
#
# Adding `G2` drops MAE from 2.18 to 0.76 (about 65% lower). It is not a better model; it is a different question. `G2` is a later grade that correlates 0.92 with `G3`, and it does not exist at enrolment, so that score is not an honest forecast of what we could know at that moment.
#
# What the lasso kept (16 columns with non-negligible weight): the largest coefficients are `school_GP` (+1.11), `failures` (−0.94) and `higher_no` (−0.79), followed by workday alcohol `Dalc` (−0.35), `schoolsup_no` (+0.36), `studytime` (+0.33), and parents' education `Medu`/`Fedu` (+0.17/+0.10). Coefficient sizes are not strictly comparable: numeric columns are standardized (effect per standard deviation), while one-hot columns are 0/1 switches. And these are associations in one sample from two schools, not causal effects. For example, a positive `schoolsup_no` coefficient may simply mean that students who receive extra support were already struggling; the lasso cannot tell us that.

# %% [markdown]
# ## Try it yourself
#
# **1. Titanic: add cabin initial and fare per family member, then compare with Part B.**
#
# Two caveats shape how to read this. `Fare` on Titanic is usually the price of a whole *ticket*, so dividing by `FamilySize` gives a rough per-person fare. And a single 25% split of 891 rows leaves only 223 validation passengers, so small AUC gaps can be noise; I add 5-fold cross-validated AUC (still with all preprocessing inside the pipeline) as a steadier second opinion.

# %%
df["CabinDeck"] = df["Cabin"].str[0].fillna("Unknown")
df["FarePerPerson"] = df["Fare"] / df["FamilySize"]
df["LogFarePerPerson"] = np.log1p(df["FarePerPerson"])

engineered_num = ["Age", "SibSp", "Parch", "LogFare", "FamilySize", "IsAlone"]
engineered_cat = ["Pclass", "Sex", "Embarked", "Title"]
variants = {
    "raw columns": (["Age", "SibSp", "Parch", "Fare"], ["Pclass", "Sex", "Embarked"]),
    "raw + engineered (Part B)": (engineered_num, engineered_cat),
    "+ cabin deck": (engineered_num, engineered_cat + ["CabinDeck"]),
    "+ fare per person": (engineered_num + ["LogFarePerPerson"], engineered_cat),
    "+ both": (engineered_num + ["LogFarePerPerson"], engineered_cat + ["CabinDeck"]),
}


def titanic_pipeline(num_cols, cat_cols):
    return Pipeline([
        ("preprocess", make_preprocess(num_cols, cat_cols)),
        ("model", LogisticRegression(max_iter=2000)),
    ])


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rows = []
for name, (num_cols, cat_cols) in variants.items():
    X = df[num_cols + cat_cols]
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    fitted = titanic_pipeline(num_cols, cat_cols).fit(X_train, y_train)
    cv_auc = cross_val_score(titanic_pipeline(num_cols, cat_cols), X, y, cv=cv, scoring="roc_auc")
    rows.append({
        "feature set": name,
        "validation AUC": roc_auc_score(y_valid, fitted.predict_proba(X_valid)[:, 1]),
        "5-fold CV AUC": cv_auc.mean(),
        "CV std": cv_auc.std(),
    })
print(pd.DataFrame(rows).set_index("feature set").round(3))

# %% [markdown]
# **Inspect.** Neither addition helps measurably. On the fixed split, AUC goes from 0.872 to 0.868 with cabin deck and stays at 0.872 with fare per person. Cross-validated AUC changes by at most 0.002 (0.873 → 0.875 / 0.873 / 0.874), too small to call a real difference given a fold-to-fold spread of about 0.02. That is plausible: `Pclass` and `LogFare` already carry most of what these two features add.

# %% [markdown]
# Is a recorded cabin just a proxy for class, or does it track survival on its own? Compare survival with and without a cabin *within* each class:

# %%
has_cabin = df["Cabin"].notna().rename("has cabin")
print(df.groupby(["Pclass", has_cabin])["Survived"].agg(["mean", "size"]).round(3))
print("\nShare of each class with a recorded cabin:\n", df.groupby("Pclass")["Cabin"].apply(lambda s: s.notna().mean()).round(3))

# %% [markdown]
# **Inspect.** Cabin also raises the lab's moment-of-prediction question. Within every class, passengers with a recorded cabin survived more often than those without: 66.5% vs 47.5% in 1st class, 81.2% vs 44.0% in 2nd, 50.0% vs 23.6% in 3rd (the 2nd- and 3rd-class groups with a cabin are tiny: 16 and 12 passengers). Cabins are also recorded very unevenly by class (81.5% of 1st class, 8.7% of 2nd, 2.4% of 3rd). That is consistent with cabin details having been recorded more completely for survivors, but this data cannot show *when* they were recorded, and that is exactly what the moment-of-prediction question needs. So `CabinDeck = "Unknown"` deserves suspicion before it goes into a real forecast, even though it did not raise accuracy here.

# %% [markdown]
# **2. Part D again with `k=5` and `k=20`.**
#
# `SelectKBest` chooses among the *encoded* columns (after one-hot encoding), so `k` is counted against a larger pool than the 30 original columns. The `k=10` row should reproduce the number from Part D.

# %%
n_encoded = make_preprocess(s_num, s_cat).fit(Xs_train).get_feature_names_out().size
print(f"{len(s_num) + len(s_cat)} original columns -> {n_encoded} encoded columns after one-hot encoding\n")

k_results = {}
for k in (5, 10, 20):
    label = f"SelectKBest, k={k}"
    k_results[label] = evaluate(Pipeline([
        ("preprocess", make_preprocess(s_num, s_cat)),
        ("select", SelectKBest(f_regression, k=k)),
        ("model", Ridge(alpha=1.0)),
    ]), Xs_train, Xs_valid, label)

comparison = pd.Series({
    "all 30 features": results["all 30 features"],
    **k_results,
    "lasso (selects itself)": results["lasso (selects itself)"],
}, name="validation MAE").round(3)
print("\n", comparison.to_string(), sep="")

# %% [markdown]
# **Inspect.** On the fixed split, validation MAE barely moves with k: 2.134 (k=5), 2.139 (k=10) and 2.125 (k=20), against 2.181 with every column and 2.077 for the lasso. The whole spread across k is 0.014 grade points, on grades that run 0–19 in this data. Note that k counts *encoded* columns, so even k=20 keeps only about a third of the 56. On this split selection looks slightly better than using everything, but a single split is a noisy judge, so the next cell repeats the comparison.

# %% [markdown]
# One split of 163 students is a noisy judge of differences this small. Repeat the whole comparison over 30 different random splits (every step is still fitted inside the pipeline on training rows only):

# %%
def student_pipeline(model, k=None):
    """Preprocess -> optional SelectKBest -> model."""
    steps = [("preprocess", make_preprocess(s_num, s_cat))]
    if k is not None:
        steps.append(("select", SelectKBest(f_regression, k=k)))
    return Pipeline(steps + [("model", model)])


candidates = {
    "all 30 features": lambda: student_pipeline(Ridge(alpha=1.0)),
    "SelectKBest, k=5": lambda: student_pipeline(Ridge(alpha=1.0), k=5),
    "SelectKBest, k=10": lambda: student_pipeline(Ridge(alpha=1.0), k=10),
    "SelectKBest, k=20": lambda: student_pipeline(Ridge(alpha=1.0), k=20),
    "lasso (selects itself)": lambda: student_pipeline(Lasso(alpha=0.1)),
}
split_maes = {name: [] for name in candidates}
for seed in range(30):
    X_tr, X_va, y_tr, y_va = train_test_split(X_s, y_s, test_size=0.25, random_state=seed)
    for name, make_model in candidates.items():
        split_maes[name].append(mean_absolute_error(y_va, make_model().fit(X_tr, y_tr).predict(X_va)))
split_maes = pd.DataFrame(split_maes)

beats_all = split_maes.drop(columns="all 30 features").lt(split_maes["all 30 features"], axis=0).mean()
print(pd.DataFrame({
    "mean MAE": split_maes.mean(),
    "std across splits": split_maes.std(),
    "share of splits beating all features": beats_all,
}).round(3))

# %% [markdown]
# **Inspect.** Over 30 random splits the picture changes. Mean MAE is 2.034 for all columns, 2.067 / 2.044 / 2.048 for k=5 / 10 / 20, and 2.028 for the lasso: differences of at most 0.04, against a split-to-split standard deviation of roughly 0.13–0.16 for every method. Selection beat the all-features model in only 33–43% of splits. So the fair conclusion is that selecting features keeps accuracy roughly unchanged while shrinking the model (k=5 costs about 0.03 grade points on average), and the lasso is as good as ridge on everything, but none of them is reliably *better*. The small edge in Part D partly reflects the luck of `random_state=42`.

# %% [markdown]
# ## Check your understanding
#
# **1. Why must train/test splitting happen before any preprocessing step that learns from the data?**
#
# Imputers, scalers, encoders and feature selectors all *learn* something from the data: medians, means and standard deviations, category lists, feature scores. If they are fitted before the split, the validation rows help decide those numbers, so information about the "unseen" data leaks into training and the validation score comes out too optimistic. Splitting first and fitting every step on the training rows only (which is what `Pipeline` + `ColumnTransformer` do: `.fit` sees `X_train`, and validation rows are only *transformed*) keeps the validation set a fair stand-in for future data. `SelectKBest` is the sharpest case, because it looks at `y`: choosing features on the full data before splitting is a direct route to inflated scores.
#
# **2. What is the difference between an identifier, a feature, and a label?**
#
# - **Label:** what we are trying to predict (`Survived`, `G3`, `diagnosis`). It is never an input.
# - **Feature:** a measurable input that is available at prediction time and can carry signal that generalizes (`Pclass`, `Age`, `studytime`). Engineered columns such as `Title` and `FamilySize` are features derived from other inputs.
# - **Identifier:** a value that names a row without describing it (`PassengerId`, `id` in the cancer data, the raw `Name` string). It carries no signal that transfers to new rows; at best it is noise, at worst it lets a flexible model memorise rows or exploit file ordering. Drop it, or extract a real feature from it, as `Title` is extracted from `Name`.
#
# The roles are set by the task, not by the column: `Name` is an identifier, but `Title` pulled out of it is a feature.
#
# **3. Why do you need to know the moment of prediction before you can say whether a feature is leakage?**
#
# Leakage is defined relative to time, not to the column itself: a feature leaks if its value would not be known at the moment the prediction is made, or if it is a consequence of the outcome. The same column can be legitimate for one forecast and leaky for another. `G2` is the example in this lab: for a forecast made at enrolment it does not exist yet, so using it (MAE 0.76 instead of 2.18) reports a score that could never be reproduced in real use. For a forecast made after the second-period grades are in, `G2` would be a perfectly valid and very valuable feature. Without fixing the moment of prediction first, there is no way to tell which features are fair. The same reasoning is why `Cabin` deserves a question mark for the Titanic model.
