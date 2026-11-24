"""
=====================================================================
 AI-BASED CROP ADVISORY RELIABILITY ASSESSMENT
 Course : Fundamentals of Data Science (DSA04)  |  Assignment Tool-5 (Hackathon)
 Student: Nandhana Sri S      Reg No: 1972260063
 Theme  : Smart Agriculture
=====================================================================
Goal
----
Build several crop-recommendation models, then check whether the best
model is RELIABLE in every kind of district (irrigated, semi-arid and
drought-prone), and decide when a human agricultural officer must
review the model's advice.

Steps in this file
------------------
 1. Create the dataset (synthetic, because no real file was supplied)
 2. Data preparation + Exploratory Data Analysis (EDA)
 3. Train 5 models and compare Accuracy and F1-score
 4. Region-wise performance (is the model fair to every region?)
 5. Unseen-district test (does it work in a district it never saw?)
 6. Bias reduction (re-train with region weights)
 7. Human-review rule (when must an officer check the advice?)
 8. Final deployment decision
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GroupKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

warnings.filterwarnings("ignore")
SEED = 42
rng = np.random.default_rng(SEED)
OUT = "charts"
os.makedirs(OUT, exist_ok=True)
sns.set_style("whitegrid")


def heading(text):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


# ---------------------------------------------------------------------
# STEP 1 : CREATE THE DATASET
# ---------------------------------------------------------------------
# 12 Tamil Nadu districts in 3 region types. Irrigated districts have the
# MOST records (high-resource) and drought-prone districts have the FEWEST
# - exactly the situation that causes bias in real life.
heading("STEP 1 : DATASET CREATION")

districts = {
    # name            : (region type,  rain mm, temp C, humidity %, records)
    "Thanjavur":        ("Irrigated",     1100, 28, 78, 700),
    "Tiruvarur":        ("Irrigated",     1200, 28, 80, 700),
    "Nagapattinam":     ("Irrigated",     1250, 29, 82, 700),
    "Erode":            ("Irrigated",      900, 29, 70, 700),
    "Coimbatore":       ("Semi-arid",      650, 29, 62, 450),
    "Salem":            ("Semi-arid",      600, 30, 60, 450),
    "Tiruvannamalai":   ("Semi-arid",      700, 30, 64, 450),
    "Ramanathapuram":   ("Drought-prone",  400, 33, 52, 150),
    "Thoothukudi":      ("Drought-prone",  380, 33, 55, 150),
    "Virudhunagar":     ("Drought-prone",  420, 32, 50, 150),
    "Dharmapuri":       ("Drought-prone",  450, 31, 54, 150),
    "Sivaganga":        ("Drought-prone",  430, 32, 53, 150),
}
soil_types = ["Clay", "Alluvial", "Black", "Red", "Sandy"]
crops = ["Rice", "Sugarcane", "Cotton", "Groundnut", "Millets", "Pulses"]

rows = []
for name, (rtype, rain, temp, hum, n) in districts.items():
    irrigated_share = {"Irrigated": 0.90, "Semi-arid": 0.45, "Drought-prone": 0.10}[rtype]
    soil_p = {
        "Irrigated":     [0.35, 0.40, 0.05, 0.10, 0.10],
        "Semi-arid":     [0.10, 0.10, 0.35, 0.35, 0.10],
        "Drought-prone": [0.03, 0.02, 0.30, 0.35, 0.30],
    }[rtype]
    # drought-prone areas have much more unstable weather (bigger spread)
    spread = {"Irrigated": 0.12, "Semi-arid": 0.20, "Drought-prone": 0.32}[rtype]
    for _ in range(n):
        rows.append({
            "district": name,
            "region_type": rtype,
            "soil_type": rng.choice(soil_types, p=soil_p),
            "soil_ph": np.clip(rng.normal(6.8, 0.7), 5.0, 8.5),
            "nitrogen": np.clip(rng.normal(70, 25), 10, 140),
            "rainfall_mm": max(100, rng.normal(rain, rain * spread)),
            "temperature_c": rng.normal(temp, 1.8 if rtype != "Drought-prone" else 2.6),
            "humidity_pct": np.clip(rng.normal(hum, 7 if rtype != "Drought-prone" else 10), 25, 95),
            "irrigation_access": int(rng.random() < irrigated_share),
            "fertilizer_kg_ha": np.clip(rng.normal(120 if rtype == "Irrigated" else 70, 35), 0, 250),
            "pest_incidents": int(rng.poisson(2 if rtype == "Irrigated" else 3)),
            "previous_crop": rng.choice(["Cereal", "Pulse", "Cash crop", "Fallow"]),
        })
df = pd.DataFrame(rows)


# The "best crop" is decided by agronomy-style scores + random noise.
def best_crop(r):
    s = {c: 0.0 for c in crops}
    s["Rice"] += (r.rainfall_mm - 700) / 500 + (r.humidity_pct - 65) / 30 \
        + 0.8 * r.irrigation_access + (0.5 if r.soil_type in ("Clay", "Alluvial") else -0.5)
    s["Sugarcane"] += 1.4 * r.irrigation_access + (r.nitrogen - 70) / 30 \
        + (0.8 if r.soil_type == "Alluvial" else 0) + (r.temperature_c - 29) / 3 \
        + (r.rainfall_mm - 800) / 500
    s["Cotton"] += (1.2 if r.soil_type == "Black" else -0.5) \
        - abs(r.rainfall_mm - 650) / 250 + (r.temperature_c - 28) / 4
    s["Groundnut"] += (1.0 if r.soil_type in ("Red", "Sandy") else -0.4) \
        - abs(r.rainfall_mm - 550) / 250 + (6.8 - r.soil_ph) / 1.0 \
        + (0.3 if r.previous_crop == "Cereal" else 0)
    s["Millets"] += (500 - r.rainfall_mm) / 150 + (r.temperature_c - 29) / 3 \
        - 1.2 * r.irrigation_access + (0.5 if r.soil_type == "Sandy" else 0) \
        + (55 - r.humidity_pct) / 20
    s["Pulses"] += (0.9 if r.previous_crop == "Cereal" else -0.2) \
        - (r.nitrogen - 50) / 40 - abs(r.rainfall_mm - 500) / 300 \
        + (0.4 if r.previous_crop == "Fallow" else 0)
    noise_sd = {"Irrigated": 0.25, "Semi-arid": 0.40, "Drought-prone": 0.85}[r.region_type]
    for c in s:
        s[c] += rng.normal(0, noise_sd)
    return max(s, key=s.get)


df["recommended_crop"] = df.apply(best_crop, axis=1)

# Add a few real-world data problems on purpose (to clean in Step 2)
for col in ["soil_ph", "nitrogen", "humidity_pct"]:
    idx = rng.choice(df.index, size=int(0.03 * len(df)), replace=False)
    df.loc[idx, col] = np.nan
dup = df.sample(40, random_state=1)
df = pd.concat([df, dup], ignore_index=True)

print(f"Rows: {df.shape[0]}   Columns: {df.shape[1]}")
print("\nFirst 5 rows:")
print(df[["district","soil_type","rainfall_mm","temperature_c","irrigation_access","recommended_crop"]].head().round(1).to_string())

# ---------------------------------------------------------------------
# STEP 2 : DATA PREPARATION + EDA
# ---------------------------------------------------------------------
heading("STEP 2 : DATA PREPARATION AND EDA")

print("Missing values BEFORE cleaning:")
print(df.isna().sum()[df.isna().sum() > 0].to_string())
print(f"\nDuplicate rows BEFORE cleaning: {df.duplicated().sum()}")

df = df.drop_duplicates().reset_index(drop=True)
for col in ["soil_ph", "nitrogen", "humidity_pct"]:
    # fill with the median of the SAME region type (fairer than one global median)
    df[col] = df[col].fillna(df.groupby("region_type")[col].transform("median"))
print(f"Missing values AFTER cleaning : {int(df.isna().sum().sum())}")
print(f"Rows AFTER removing duplicates: {len(df)}")

print("\nRecords per region type (note the imbalance):")
print(df["region_type"].value_counts().to_string())

print("\nCrop distribution (%) by region type:")
print((pd.crosstab(df["region_type"], df["recommended_crop"], normalize="index") * 100).round(1).to_string())

print("\nAverage climate by region type:")
print(df.groupby("region_type")[["rainfall_mm", "temperature_c", "humidity_pct",
                                 "irrigation_access", "fertilizer_kg_ha"]].mean().round(1).to_string())

fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
df["region_type"].value_counts().plot(kind="bar", ax=ax[0], color=["#2e7d32", "#f9a825", "#c62828"])
ax[0].set_title("Records per region type"); ax[0].set_xlabel(""); ax[0].tick_params(axis="x", rotation=0)
pd.crosstab(df["region_type"], df["recommended_crop"], normalize="index").plot(
    kind="bar", stacked=True, ax=ax[1], colormap="Set2")
ax[1].set_title("Crop mix by region type"); ax[1].set_xlabel(""); ax[1].tick_params(axis="x", rotation=0)
ax[1].legend(fontsize=7, loc="upper right")
sns.boxplot(data=df, x="region_type", y="rainfall_mm", ax=ax[2],
            palette=["#2e7d32", "#f9a825", "#c62828"])
ax[2].set_title("Rainfall by region type"); ax[2].set_xlabel("")
plt.tight_layout(); plt.savefig(f"{OUT}/01_eda.png", dpi=130); plt.close()

# ---------------------------------------------------------------------
# STEP 3 : TRAIN AND COMPARE MODELS
# ---------------------------------------------------------------------
heading("STEP 3 : MODEL TRAINING AND COMPARISON")

num_cols = ["soil_ph", "nitrogen", "rainfall_mm", "temperature_c", "humidity_pct",
            "fertilizer_kg_ha", "pest_incidents", "irrigation_access"]
cat_cols = ["soil_type", "previous_crop"]
features = num_cols + cat_cols
# NOTE: district and region_type are NOT used as inputs. They are only used
# later to CHECK fairness.

X = df[features]
y = df["recommended_crop"]
strat = df["region_type"]
X_tr, X_te, y_tr, y_te, meta_tr, meta_te = train_test_split(
    X, y, df[["district", "region_type"]], test_size=0.25, random_state=SEED, stratify=strat)
print(f"Training rows: {len(X_tr)}   Testing rows: {len(X_te)}")


def make_pipe(model):
    prep = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)])
    return Pipeline([("prep", prep), ("model", model)])


models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Decision Tree":       DecisionTreeClassifier(max_depth=8, random_state=SEED),
    "KNN":                 KNeighborsClassifier(n_neighbors=11),
    "Random Forest":       RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
    "Gradient Boosting":   GradientBoostingClassifier(random_state=SEED),
}

results, fitted, preds = [], {}, {}
for name, mdl in models.items():
    pipe = make_pipe(mdl).fit(X_tr, y_tr)
    p = pipe.predict(X_te)
    fitted[name], preds[name] = pipe, p
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_te, p),
        "Macro F1": f1_score(y_te, p, average="macro"),
        "Weighted F1": f1_score(y_te, p, average="weighted"),
    })
res = pd.DataFrame(results).sort_values("Macro F1", ascending=False).reset_index(drop=True)
print(res.round(4).to_string(index=False))
best_name = res.loc[0, "Model"]
print(f"\nBest overall model (highest Macro F1): {best_name}")

fig, ax = plt.subplots(figsize=(8, 4.2))
res.set_index("Model")[["Accuracy", "Macro F1"]].plot(kind="bar", ax=ax, color=["#1565c0", "#ef6c00"])
ax.set_ylim(0.5, 1.0); ax.set_title("Model comparison (test set)"); ax.set_xlabel("")
ax.tick_params(axis="x", rotation=20)
plt.tight_layout(); plt.savefig(f"{OUT}/02_model_comparison.png", dpi=130); plt.close()

# ---------------------------------------------------------------------
# STEP 4 : REGION-WISE PERFORMANCE
# ---------------------------------------------------------------------
heading("STEP 4 : REGION-WISE PERFORMANCE (BIAS CHECK)")

region_order = ["Irrigated", "Semi-arid", "Drought-prone"]
rw = []
for name in models:
    for reg in region_order:
        m = (meta_te["region_type"] == reg).values
        rw.append({"Model": name, "Region": reg,
                   "Accuracy": accuracy_score(y_te[m], preds[name][m]),
                   "Macro F1": f1_score(y_te[m], preds[name][m], average="weighted")})
rw = pd.DataFrame(rw)
print("Accuracy by region:")
print(rw.pivot(index="Model", columns="Region", values="Accuracy")[region_order].round(3).to_string())
print("\nF1-score (weighted) by region:")
f1_tab = rw.pivot(index="Model", columns="Region", values="Macro F1")[region_order]
print(f1_tab.round(3).to_string())
f1_tab["Gap (Irrigated - Drought)"] = f1_tab["Irrigated"] - f1_tab["Drought-prone"]
print("\nBias gap (Irrigated F1 minus Drought-prone F1)  -> bigger = more biased:")
print(f1_tab["Gap (Irrigated - Drought)"].round(3).to_string())

fig, ax = plt.subplots(figsize=(8.5, 4.6))
sns.heatmap(f1_tab[region_order], annot=True, fmt=".3f", cmap="RdYlGn", vmin=0.55, vmax=0.95, ax=ax)
ax.set_title("Region-wise F1-score (weighted)"); ax.set_ylabel("")
plt.tight_layout(); plt.savefig(f"{OUT}/03_regionwise_f1.png", dpi=130); plt.close()

# Confusion matrix of best model in the weakest region
m_dr = (meta_te["region_type"] == "Drought-prone").values
cm = confusion_matrix(y_te[m_dr], preds[best_name][m_dr], labels=crops)
fig, ax = plt.subplots(figsize=(6, 4.8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=crops, yticklabels=crops, ax=ax)
ax.set_xlabel("Predicted crop"); ax.set_ylabel("Actual crop")
ax.set_title(f"{best_name}: Drought-prone districts")
plt.tight_layout(); plt.savefig(f"{OUT}/04_confusion_drought.png", dpi=130); plt.close()

# ---------------------------------------------------------------------
# STEP 5 : UNSEEN-DISTRICT TEST
# ---------------------------------------------------------------------
heading("STEP 5 : UNSEEN-DISTRICT TEST (GroupKFold by district)")
print("Each district is held out completely, then predicted by a model that never saw it.\n")

gkf = GroupKFold(n_splits=6)
p_cv = cross_val_predict(make_pipe(models[best_name].__class__(**models[best_name].get_params())),
                         X, y, groups=df["district"], cv=gkf)
cvdf = df[["district", "region_type"]].copy()
cvdf["ok"] = (p_cv == y.values)
dist_acc = cvdf.groupby(["region_type", "district"])["ok"].mean().reset_index()
dist_acc["ok"] = dist_acc["ok"].round(3)
dist_acc["region_type"] = pd.Categorical(dist_acc["region_type"], region_order, ordered=True)
dist_acc = dist_acc.sort_values(["region_type", "ok"], ascending=[True, False])
dist_acc.columns = ["Region type", "District", "Accuracy on unseen district"]
print(dist_acc.to_string(index=False))
print(f"\nOverall unseen-district accuracy: {cvdf['ok'].mean():.3f}")
print(f"Overall unseen-district Macro F1: {f1_score(y, p_cv, average='macro'):.3f}")

fig, ax = plt.subplots(figsize=(9, 4.6))
colors = dist_acc["Region type"].map({"Irrigated": "#2e7d32", "Semi-arid": "#f9a825",
                                      "Drought-prone": "#c62828"}).astype(str)
ax.bar(dist_acc["District"], dist_acc["Accuracy on unseen district"], color=colors)
ax.set_ylim(0.5, 1.0); ax.set_title("Accuracy on unseen districts"); ax.tick_params(axis="x", rotation=60)
plt.tight_layout(); plt.savefig(f"{OUT}/05_unseen_districts.png", dpi=130); plt.close()

# ---------------------------------------------------------------------
# STEP 6 : BIAS REDUCTION (REGION-BALANCED TRAINING)
# ---------------------------------------------------------------------
heading("STEP 6 : BIAS REDUCTION USING REGION WEIGHTS")
print("Idea: give drought-prone records MORE importance during training so the")
print("model does not copy only the pattern of the large irrigated regions.\n")

freq = meta_tr["region_type"].value_counts()
w = meta_tr["region_type"].map(lambda r: len(meta_tr) / (3 * freq[r])).values
print("Weight per region:", {k: round(len(meta_tr) / (3 * v), 2) for k, v in freq.items()})

if best_name == "KNN":          # KNN cannot take sample weights, so use Random Forest
    bal_model = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)
else:
    bal_model = models[best_name].__class__(**models[best_name].get_params())
bal = make_pipe(bal_model)
bal.fit(X_tr, y_tr, model__sample_weight=w)
p_bal = bal.predict(X_te)
proba_bal = bal.predict_proba(X_te)

rows_b = []
for reg in region_order:
    m = (meta_te["region_type"] == reg).values
    rows_b.append({"Region": reg,
                   "F1 before": f1_score(y_te[m], preds[best_name][m], average="weighted"),
                   "F1 after": f1_score(y_te[m], p_bal[m], average="weighted"),
                   "Acc before": accuracy_score(y_te[m], preds[best_name][m]),
                   "Acc after": accuracy_score(y_te[m], p_bal[m])})
cmp_b = pd.DataFrame(rows_b)
print(cmp_b.round(3).to_string(index=False))
print(f"\nOverall Accuracy : {accuracy_score(y_te, preds[best_name]):.4f} -> {accuracy_score(y_te, p_bal):.4f}")
print(f"Overall Macro F1 : {f1_score(y_te, preds[best_name], average='macro'):.4f} -> "
      f"{f1_score(y_te, p_bal, average='macro'):.4f}")
gap_b = cmp_b.loc[0, "F1 before"] - cmp_b.loc[2, "F1 before"]
gap_a = cmp_b.loc[0, "F1 after"] - cmp_b.loc[2, "F1 after"]
print(f"Bias gap (Irrigated - Drought F1): {gap_b:.3f} -> {gap_a:.3f}")

acc_drop = accuracy_score(y_te, preds[best_name]) - accuracy_score(y_te, p_bal)
use_balanced = (abs(gap_a) < abs(gap_b)) and acc_drop < 0.02
print(f"Fairer gap? {abs(gap_a) < abs(gap_b)}   Accuracy drop: {acc_drop:.4f} (allowed < 0.02)")
final_model = bal if use_balanced else fitted[best_name]
final_proba = proba_bal if use_balanced else fitted[best_name].predict_proba(X_te)
final_pred = p_bal if use_balanced else preds[best_name]
final_label = f"{best_name} (region-balanced)" if use_balanced else best_name
print(f"\nModel chosen for next steps: {final_label}")

# Feature importance
core = final_model.named_steps["model"]
names = num_cols + list(final_model.named_steps["prep"].named_transformers_["cat"].get_feature_names_out(cat_cols))
if hasattr(core, "feature_importances_"):
    vals = core.feature_importances_
else:                                   # Logistic Regression: use average size of coefficients
    vals = np.abs(core.coef_).mean(axis=0)
    vals = vals / vals.sum()
imp = pd.Series(vals, index=names).sort_values(ascending=False)
print("\nTop 8 features that drive the advice:")
print(imp.head(8).round(3).to_string())
fig, ax = plt.subplots(figsize=(7, 4.2))
imp.head(10)[::-1].plot(kind="barh", ax=ax, color="#2e7d32")
ax.set_title("Top 10 feature importances"); plt.tight_layout()
plt.savefig(f"{OUT}/06_feature_importance.png", dpi=130); plt.close()

# ---------------------------------------------------------------------
# STEP 7 : HUMAN-REVIEW RULE
# ---------------------------------------------------------------------
heading("STEP 7 : WHEN MUST A HUMAN OFFICER REVIEW THE ADVICE?")

conf = final_proba.max(axis=1)
correct = (final_pred == y_te.values)
train_rain = X_tr["rainfall_mm"]
z_rain = ((X_te["rainfall_mm"] - train_rain.mean()) / train_rain.std()).values

print("Confidence threshold sweep (model's own confidence in its top crop):")
sweep = []
for t in [0.40, 0.50, 0.60, 0.70, 0.80]:
    auto = conf >= t
    sweep.append({"Threshold": t,
                  "Auto-approved %": 100 * auto.mean(),
                  "Accuracy of auto-approved": correct[auto].mean(),
                  "Sent to officer %": 100 * (~auto).mean()})
sweep = pd.DataFrame(sweep)
print(sweep.round(3).to_string(index=False))

THRESH = 0.60
flag_low_conf = conf < THRESH
flag_extreme_rain = np.abs(z_rain) > 2.0
flag_drought_unsure = (meta_te["region_type"].values == "Drought-prone") & (conf < 0.75)
review = flag_low_conf | flag_extreme_rain | flag_drought_unsure

print(f"\nFINAL RULE: officer reviews the advice if ANY of these is true:")
print(f"  (a) model confidence < {THRESH:.0%}                      -> {flag_low_conf.sum()} cases")
print(f"  (b) rainfall is extreme (|z-score| > 2 vs training)  -> {flag_extreme_rain.sum()} cases")
print(f"  (c) drought-prone district AND confidence < 75%      -> {flag_drought_unsure.sum()} cases")
print(f"\nTotal sent to officer : {review.sum()} of {len(review)} ({100 * review.mean():.1f}%)")
print(f"Accuracy WITHOUT review rule (all cases)      : {correct.mean():.3f}")
print(f"Accuracy of AUTO-APPROVED cases               : {correct[~review].mean():.3f}")
print(f"Accuracy of cases SENT to officer (risky ones): {correct[review].mean():.3f}")

print("\nReview rate and auto-approved accuracy by region:")
rv = []
for reg in region_order:
    m = (meta_te["region_type"] == reg).values
    rv.append({"Region": reg, "Review rate %": 100 * review[m].mean(),
               "Auto-approved accuracy": correct[m & ~review].mean()})
print(pd.DataFrame(rv).round(3).to_string(index=False))

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].hist(conf[correct], bins=20, alpha=0.7, label="Correct", color="#2e7d32")
ax[0].hist(conf[~correct], bins=20, alpha=0.7, label="Wrong", color="#c62828")
ax[0].axvline(THRESH, ls="--", color="k"); ax[0].legend()
ax[0].set_title("Model confidence: correct vs wrong"); ax[0].set_xlabel("Confidence")
ax[1].plot(sweep["Auto-approved %"], sweep["Accuracy of auto-approved"], marker="o", color="#1565c0")
for _, r in sweep.iterrows():
    ax[1].annotate(f"{r['Threshold']:.2f}", (r["Auto-approved %"], r["Accuracy of auto-approved"]),
                   textcoords="offset points", xytext=(4, 5), fontsize=8)
ax[1].set_xlabel("% advice auto-approved"); ax[1].set_ylabel("Accuracy of auto-approved")
ax[1].set_title("Trade-off: automation vs accuracy")
plt.tight_layout(); plt.savefig(f"{OUT}/07_human_review.png", dpi=130); plt.close()

# ---------------------------------------------------------------------
# STEP 8 : FINAL DECISION
# ---------------------------------------------------------------------
heading("STEP 8 : FINAL DEPLOYMENT RECOMMENDATION")
print(f"1. Deploy model        : {final_label}")
print(f"2. Overall test result : Accuracy {accuracy_score(y_te, final_pred):.3f}, "
      f"Macro F1 {f1_score(y_te, final_pred, average='macro'):.3f}")
for reg in region_order:
    m = (meta_te["region_type"] == reg).values
    print(f"   {reg:<14}: Accuracy {accuracy_score(y_te[m], final_pred[m]):.3f}, "
          f"Weighted F1 {f1_score(y_te[m], final_pred[m], average='weighted'):.3f}")
print("3. Human review needed when: confidence < 60%, OR extreme rainfall,")
print("   OR drought-prone district with confidence < 75%.")
print("4. Re-train every season and keep collecting more drought-prone data.")
print("5. Publish region-wise accuracy to officers so that weaker areas get extra supervision.")

# save final test predictions for submission proof
out = meta_te.copy()
out["actual_crop"] = y_te.values
out["predicted_crop"] = final_pred
out["confidence"] = conf.round(3)
out["officer_review"] = np.where(review, "YES", "NO")
out.to_csv("final_predictions.csv", index=False)
df.to_csv("crop_advisory_dataset.csv", index=False)
print("\nSaved: crop_advisory_dataset.csv, final_predictions.csv and charts/*.png")
