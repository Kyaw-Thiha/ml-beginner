import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    classification_report,
)


# |%%--%%| <bcq3B9MeVR|LmEKTd8qPu>

df = pd.read_csv("iris/Iris.csv")

# Quick Analysis
print("Head")
print(df.head)
print("------------------------\n")

print("Shape")
print("Rows: ", df.shape[0])
print("Columns: ", df.shape[1])
print("------------------------\n")

print("Columns")
for col, dtype in zip(df.columns, df.dtypes):
    print(col, " ----- ", dtype)
print("------------------------\n")

# |%%--%%| <LmEKTd8qPu|MDO9yoyw7r>

# Handling missing values
df.isnull().sum()  # Missing values per column
df.dropna()  # Drop rows with missing values
df.fillna(0)  # Replace missing values with 0

# |%%--%%| <MDO9yoyw7r|TB8RkJClUA>

# Analyzing unique values

# print(df["SepalLengthCm"].value_counts())  # Count of unique values
# print(df["SepalLengthCm"].unique())  # Unique values
# print(df["SepalLengthCm"].nunique())  # Number of unique values

for col in df.columns:
    print(col, "---", df[col].nunique())


# |%%--%%| <TB8RkJClUA|0mM04Nm58O>

# Statistical Analysis

print("Summary")
print(df.describe())
print("------------------------\n")

print("Mean")
print(df.mean(numeric_only=True))
print("------------------------\n")

print("Median")
print(df.median(numeric_only=True))
print("------------------------\n")

print("Standard Deviation")
print(df.std(numeric_only=True))
print("------------------------\n")

print("Maximum")
print(df.max(numeric_only=True))
print("------------------------\n")

print("Minimum")
print(df.min(numeric_only=True))
print("------------------------\n")

print("Range")
print(df.max(numeric_only=True) - df.min(numeric_only=True))
print("------------------------\n")


# |%%--%%| <0mM04Nm58O|saHlaJx1BG>

# Basic Plotting

# Excluding non-numeric values, and columns with all NaN
plot_df = df.select_dtypes(include="number").dropna(axis=1, how="all")

for col in plot_df.columns:
    df.plot(y=col)


# print("Box Plot")
# df.boxplot(column=[col for col in plot_df.columns])

# |%%--%%| <saHlaJx1BG|vYq0K12UKJ>

# Scatter plots

print("Scatter Plots")
sns.pairplot(df.drop(columns="Id"), hue="Species")
plt.suptitle("Pairplot of Numeric Columns", y=1.02)
plt.show()


# |%%--%%| <vYq0K12UKJ|5dlvkuiavB>

# Correlation Heat Map
plt.figure(figsize=(10, 6))
sns.heatmap(df.drop(columns="Id").corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.show()


# |%%--%%| <5dlvkuiavB|yf4JuvzRqX>


# Swarm Plot
sample_num = 50

for col in df:
    if col != "Species" and col != "Id":
        plt.figure(figsize=(6, 4))
        sns.swarmplot(x="Species", y=col, data=df.sample(sample_num))
        plt.title(f"Swarmplot of {col}")
        plt.tight_layout()
        plt.show()


# |%%--%%| <yf4JuvzRqX|WntdCYYY3B>


# Violin Plot
for col in df:
    if col != "Species" and col != "Id":
        plt.figure(figsize=(6, 4))
        sns.violinplot(x="Species", y=col, data=df)
        plt.title(f"Swarmplot of {col}")
        plt.tight_layout()
        plt.show()

# |%%--%%| <WntdCYYY3B|zWeJnJ2cdW>

# Count Plot
sns.countplot(x="Species", data=df)


# |%%--%%| <zWeJnJ2cdW|ht0OTCy3pH>

# Kernel Density Plot
for col in plot_df:
    if col != "Species" and col != "Id":
        plt.figure(figsize=(6, 4))
        sns.kdeplot(data=plot_df, x=col, label=col, fill=True)
        plt.title(f"Swarmplot of {col}")
        plt.tight_layout()
        plt.show()


# |%%--%%| <ht0OTCy3pH|oac2FyCeoN>

# Regression Line
sns.pairplot(df, kind="reg", hue="Species")
plt.show()

# |%%--%%| <oac2FyCeoN|xjHspT486N>

le = LabelEncoder()

X = df[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]
y = le.fit_transform(df["Species"])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# |%%--%%| <xjHspT486N|8FyfDMoTSa>

# KNN Classification
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print("KNN Accuracy:", accuracy_score(y_test, y_pred))

# |%%--%%| <8FyfDMoTSa|JoQzOUNIeX>

# Logisitc Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(
    "Classification Report:\n",
    classification_report(y_test, y_pred, target_names=le.classes_),
)

# |%%--%%| <JoQzOUNIeX|CDDtZzzs5N>

# Naive-Bayes
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(
    "Classification Report:\n",
    classification_report(y_test, y_pred, target_names=le.classes_),
)

# |%%--%%| <CDDtZzzs5N|z9sUofqroB>

# Grid Search of RandomForestClassifier

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [2, 3, 5, None],
    "min_samples_split": [2, 5, 10],
}
grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
grid.fit(X, y)
print("Best parameters:", grid.best_params_)
print("Best score (CV):", grid.best_score_)


# |%%--%%| <z9sUofqroB|Y2ywfX1vct>

# Random Forest
model = RandomForestClassifier(
    n_estimators=50, max_depth=3, min_samples_split=2, random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(
    "Classification Report:\n",
    classification_report(y_test, y_pred, target_names=le.classes_),
)

# |%%--%%| <Y2ywfX1vct|oWHq2gw5rV>

# Grid Search of RandomForestClassifier

param_grid = {
    "learning_rate": [0.01, 0.1],
    "max_iter": [100, 200],
    "max_depth": [3, 5, None],
    "min_samples_leaf": [10, 20],
    "l2_regularization": [0, 1, 10],
    "max_leaf_nodes": [15, 31],
}
grid = GridSearchCV(
    HistGradientBoostingClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
)
grid.fit(X, y)
print("Best parameters:", grid.best_params_)
print("Best score (CV):", grid.best_score_)


# |%%--%%| <oWHq2gw5rV|pYLqzLmAmP>

# Gradient Boosted Trees
model = HistGradientBoostingClassifier(
    random_state=42,
    l2_regularization=0,
    learning_rate=0.1,
    max_depth=5,
    max_iter=100,
    max_leaf_nodes=15,
    min_samples_leaf=10,
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(
    "Classification Report:\n",
    classification_report(y_test, y_pred, target_names=le.classes_),
)


# |%%--%%| <pYLqzLmAmP|vPuIdGkLfL>

# Visualizing Results
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, stratify=y, test_size=0.2, random_state=42
)


def plot_decision_boundary(clf, X, y, title):
    h = 0.01  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    clf.fit(X, y)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(6, 4))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="Accent")
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap="Accent", edgecolor="k")
    plt.title(title)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.tight_layout()
    plt.show()


# Instantiate models
models = {
    "Logistic Regression": LogisticRegression(),
    "KNN": KNeighborsClassifier(n_neighbors=3),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(
        n_estimators=50, max_depth=3, min_samples_split=2, random_state=42
    ),
    "HistGradientBoosting": HistGradientBoostingClassifier(
        random_state=42,
        l2_regularization=0,
        learning_rate=0.1,
        max_depth=5,
        max_iter=100,
        max_leaf_nodes=15,
        min_samples_leaf=10,
    ),
}

# Plot all
for name, model in models.items():
    plot_decision_boundary(model, X_train, y_train, title=f"{name} (Train)")


# |%%--%%| <vPuIdGkLfL|eWZwS0qoLq>
