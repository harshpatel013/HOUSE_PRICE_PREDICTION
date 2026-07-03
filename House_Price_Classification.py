
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing   import StandardScaler, LabelEncoder
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import RandomForestClassifier
from sklearn.metrics         import (accuracy_score, confusion_matrix,
                                     classification_report)

plt.rcParams["figure.dpi"] = 110
sns.set_style("whitegrid")

print("Libraries imported successfully!\n")


# Load the dataset
df = pd.read_csv("train.csv")

print(f"Dataset shape  : {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Memory usage   : {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
print("\nSample rows:")
print(df[["OverallQual", "GrLivArea", "YearBuilt",
          "GarageCars", "TotalBsmtSF", "SalePrice"]].head())

# Create Price Category  (our target label)
# We divide houses into 3 price buckets using percentiles
low_cut    = df["SalePrice"].quantile(0.34)
high_cut   = df["SalePrice"].quantile(0.67)

def price_label(price):
    if price <= low_cut:
        return "Low"
    elif price <= high_cut:
        return "Medium"
    else:
        return "High"

df["PriceClass"] = df["SalePrice"].apply(price_label)

print(f"\nPrice boundaries:")
print(f"  Low    → below  ${low_cut:,.0f}")
print(f"  Medium → ${low_cut:,.0f} – ${high_cut:,.0f}")
print(f"  High   → above  ${high_cut:,.0f}")
print("\nClass distribution:")
print(df["PriceClass"].value_counts())

# Exploratory Data Analysis  (EDA)
print("\n--- EDA ---")
print("Missing values (top 10):")
missing = df.isnull().sum().sort_values(ascending=False)
print(missing[missing > 0].head(10).to_string())

fig, axes = plt.subplots(2, 3, figsize=(17, 10))
fig.suptitle("House Price Prediction — EDA", fontsize=16, fontweight="bold", y=1.01)

palette = {"Low": "#e74c3c", "Medium": "#f39c12", "High": "#27ae60"}

# SalePrice histogram
axes[0,0].hist(df["SalePrice"], bins=45, color="#3498db",
               edgecolor="white", alpha=0.85)
axes[0,0].axvline(low_cut,  color="#e74c3c",  linestyle="--", linewidth=1.8, label="Low/Med boundary")
axes[0,0].axvline(high_cut, color="#27ae60",  linestyle="--", linewidth=1.8, label="Med/High boundary")
axes[0,0].set_title("Sale Price Distribution", fontweight="bold")
axes[0,0].set_xlabel("Price ($)")
axes[0,0].set_ylabel("Count")
axes[0,0].legend(fontsize=8)

# Class count bar
order   = ["Low", "Medium", "High"]
counts  = [df["PriceClass"].value_counts()[c] for c in order]
colors  = [palette[c] for c in order]
bars    = axes[0,1].bar(order, counts, color=colors, edgecolor="black", width=0.5)
axes[0,1].set_title("Houses per Price Class", fontweight="bold")
axes[0,1].set_ylabel("Count")
for bar, cnt in zip(bars, counts):
    axes[0,1].text(bar.get_x() + bar.get_width()/2,
                   bar.get_height() + 8, str(cnt),
                   ha="center", fontsize=11, fontweight="bold")

# Overall Quality vs Price
qual_avg = df.groupby("OverallQual")["SalePrice"].mean() / 1000
axes[0,2].bar(qual_avg.index, qual_avg.values,
              color="#9b59b6", edgecolor="black", alpha=0.85)
axes[0,2].set_title("Avg Price by Overall Quality", fontweight="bold")
axes[0,2].set_xlabel("Overall Quality (1 = Poor, 10 = Excellent)")
axes[0,2].set_ylabel("Avg Sale Price ($ thousands)")

# Living Area vs Price (scatter)
for cls in order:
    subset = df[df["PriceClass"] == cls]
    axes[1,0].scatter(subset["GrLivArea"], subset["SalePrice"],
                      c=palette[cls], label=cls, alpha=0.45, s=18)
axes[1,0].set_title("Living Area vs Sale Price", fontweight="bold")
axes[1,0].set_xlabel("Above Ground Living Area (sq ft)")
axes[1,0].set_ylabel("Sale Price ($)")
axes[1,0].legend(title="Price Class")

# Year Built vs Avg Price
yr = df.groupby("YearBuilt")["SalePrice"].mean()
axes[1,1].plot(yr.index, yr.values / 1000, color="#1abc9c", linewidth=2)
axes[1,1].fill_between(yr.index, yr.values / 1000, alpha=0.2, color="#1abc9c")
axes[1,1].set_title("Year Built vs Avg Sale Price", fontweight="bold")
axes[1,1].set_xlabel("Year Built")
axes[1,1].set_ylabel("Avg Price ($ thousands)")

# Correlation heatmap
key_cols = ["SalePrice","OverallQual","GrLivArea","GarageCars",
            "TotalBsmtSF","FullBath","YearBuilt","TotRmsAbvGrd"]
sns.heatmap(df[key_cols].corr(), ax=axes[1,2], annot=True,
            fmt=".2f", cmap="coolwarm", linewidths=0.4,
            annot_kws={"size": 8})
axes[1,2].set_title("Correlation Heatmap", fontweight="bold")

plt.tight_layout()
plt.savefig("eda_plots.png", bbox_inches="tight")

# Feature Selection & Preprocessing

# Picked features that make intuitive sense for house prices
selected_features = [
    "OverallQual",   # build quality rating
    "GrLivArea",     # above-ground living area
    "GarageCars",    # garage capacity
    "GarageArea",    # garage size in sq ft
    "TotalBsmtSF",   # total basement area
    "1stFlrSF",      # first floor area
    "FullBath",      # full bathrooms
    "TotRmsAbvGrd",  # total rooms above ground
    "YearBuilt",     # year of construction
    "YearRemodAdd",  # year of last remodel
    "Fireplaces",    # number of fireplaces
    "LotArea",       # lot size
    "BsmtFinSF1",    # finished basement area
    "OpenPorchSF",   # porch area
    "WoodDeckSF",    # wood deck area
    "Neighborhood",  # location matters a lot!
    "BldgType",      # single/multi family
    "HouseStyle",    # 1-story, 2-story, etc.
    "CentralAir",    # yes/no central air
    "KitchenQual",   # kitchen quality
]

data = df[selected_features + ["PriceClass"]].copy()

# Fix missing values
for col in data.select_dtypes(include="number").columns:
    n_miss = data[col].isnull().sum()
    if n_miss > 0:
        data[col] = data[col].fillna(data[col].median())
        print(f"Filled {n_miss} missing in '{col}' with median")

# Encode text columns as numbers
le = LabelEncoder()
cat_cols = data.select_dtypes(exclude="number").columns.tolist()
cat_cols.remove("PriceClass")
for col in cat_cols:
    data[col] = le.fit_transform(data[col].astype(str))
print(f"Label-encoded: {cat_cols}")

# Feature engineering — house age is more useful than raw year
data["HouseAge"]        = 2025 - data["YearBuilt"]
data["RemodAge"]        = 2025 - data["YearRemodAdd"]
data["TotalSF"]         = data["TotalBsmtSF"] + data["1stFlrSF"] + data["GrLivArea"]
data["BathPerRoom"]     = data["FullBath"] / (data["TotRmsAbvGrd"] + 1)

print("New features created: HouseAge, RemodAge, TotalSF, BathPerRoom")

# Encode target  Low=0  Medium=1  High=2
label_map = {"Low": 0, "Medium": 1, "High": 2}
y = data["PriceClass"].map(label_map)
X = data.drop(columns=["PriceClass", "YearBuilt", "YearRemodAdd"])

print(f"\nFinal feature count : {X.shape[1]}")
print(f"Total samples       : {len(X)}")

# Train / Test split  →  80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=7, stratify=y
)

# Scale features (important for KNN)
scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"\nTrain samples : {X_train.shape[0]}")
print(f"Test  samples : {X_test.shape[0]}")

# Train Models
print("\n--- Training Models ---")

# K-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=7, metric="euclidean")
knn.fit(X_train_sc, y_train)
print("KNN trained  (k=7)")

# Decision Tree
dt = DecisionTreeClassifier(max_depth=7, min_samples_leaf=8, random_state=7)
dt.fit(X_train, y_train)
print("Decision Tree trained  (max_depth=7)")

# Random Forest
rf = RandomForestClassifier(n_estimators=120, max_depth=9,
                             min_samples_leaf=5, random_state=7)
rf.fit(X_train, y_train)
print("Random Forest trained  (120 trees)")

# Evaluate Models
print("\n--- Model Results ---")
class_labels = ["Low", "Medium", "High"]

results = {}
for name, model, Xtr, Xte in [
    ("KNN",             knn, X_train_sc, X_test_sc),
    ("Decision Tree",   dt,  X_train,    X_test),
    ("Random Forest",   rf,  X_train,    X_test),
]:
    y_pred = model.predict(Xte)
    acc    = accuracy_score(y_test, y_pred)
    cv     = cross_val_score(model, Xtr, y_train, cv=5, scoring="accuracy").mean()
    results[name] = {"acc": acc, "cv": cv, "pred": y_pred}

    print(f"\n{'='*45}")
    print(f"  {name}")
    print(f"{'='*45}")
    print(f"  Test Accuracy     : {acc*100:.2f}%")
    print(f"  5-Fold CV Avg     : {cv*100:.2f}%")
    print(f"\n  Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    print(f"\n  Classification Report:\n"
          f"{classification_report(y_test, y_pred, target_names=class_labels)}")

# Visualise Results
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Model Evaluation", fontsize=14, fontweight="bold")

# Accuracy comparison
names = list(results.keys())
test_accs = [results[m]["acc"]*100 for m in names]
cv_accs   = [results[m]["cv"]*100  for m in names]
x = np.arange(len(names))
w = 0.35
b1 = axes[0].bar(x - w/2, test_accs, width=w, label="Test Accuracy",
                  color="#3498db", edgecolor="black")
b2 = axes[0].bar(x + w/2, cv_accs,   width=w, label="CV Accuracy",
                  color="#e67e22", edgecolor="black")
axes[0].set_xticks(x); axes[0].set_xticklabels(names)
axes[0].set_ylim(50, 100)
axes[0].set_title("Test vs Cross-Validation Accuracy")
axes[0].set_ylabel("Accuracy (%)")
axes[0].legend()
for bar in list(b1) + list(b2):
    axes[0].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.4,
                 f"{bar.get_height():.1f}%",
                 ha="center", fontsize=8, fontweight="bold")

# Best model confusion matrix
best_name = max(results, key=lambda m: results[m]["acc"])
best_cm   = confusion_matrix(y_test, results[best_name]["pred"])
sns.heatmap(best_cm, annot=True, fmt="d", cmap="Blues", ax=axes[1],
            xticklabels=class_labels, yticklabels=class_labels, linewidths=0.5)
axes[1].set_title(f"Confusion Matrix — {best_name} (Best)", fontweight="bold")
axes[1].set_xlabel("Predicted Label")
axes[1].set_ylabel("True Label")

# Feature importance from Random Forest
feat_imp = pd.Series(rf.feature_importances_, index=X.columns)
feat_imp = feat_imp.sort_values(ascending=True).tail(14)
axes[2].barh(feat_imp.index, feat_imp.values,
             color="#27ae60", edgecolor="black", alpha=0.85)
axes[2].set_title("Feature Importance (Random Forest)", fontweight="bold")
axes[2].set_xlabel("Importance Score")

plt.tight_layout()
plt.savefig("model_results.png", bbox_inches="tight")
plt.close()
print("\n[Saved] model_results.png")

# Confusion matrices for all 3 models
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
fig.suptitle("Confusion Matrices — All Models", fontsize=13, fontweight="bold")
for ax, name in zip(axes, results):
    cm = confusion_matrix(y_test, results[name]["pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd", ax=ax,
                xticklabels=class_labels, yticklabels=class_labels)
    ax.set_title(name, fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")

plt.tight_layout()
plt.savefig("all_confusion_matrices.png", bbox_inches="tight")
plt.close()
print("[Saved] all_confusion_matrices.png")

# Summary
print("\n" + "="*50)
print("  FINAL SUMMARY")
print("="*50)
print(f"\n  {'Model':<18} {'Test Acc':>10}  {'CV Acc':>8}")
print(f"  {'-'*40}")
for name, res in results.items():
    print(f"  {name:<18} {res['acc']*100:>9.2f}%  {res['cv']*100:>7.2f}%")

print(f"\n  Best Model: {best_name}  ({results[best_name]['acc']*100:.2f}%)")

print("="*50)