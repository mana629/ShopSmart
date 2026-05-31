# 🛒 ShopSmart — Purchase Prediction using Decision Tree

> Supervised Machine Learning · Assignment 4  
> Predicting whether an e-commerce visitor will make a purchase based on session behaviour

---

## 📌 Problem Statement

**ShopSmart** is an e-commerce company that cannot accurately predict which visitors are likely to complete a purchase — leading to inefficient marketing and lost revenue.

Using data from **12,330 unique user sessions**, the goal is to build a machine learning model that predicts purchase intent from browsing behaviour.

---

## 🎯 Objective

- Perform **Exploratory Data Analysis (EDA)** on session data
- Apply **feature preprocessing and transformation**
- Train a **Decision Tree Classifier** (baseline)
- **Prune** the tree using `GridSearchCV` to improve generalisation
- Evaluate using **F1 Score** (benchmark: ≥ 0.55)

---

## 📁 Project Structure

```
ShopSmart-ML/
│
├── _SHOPSMART.ipynb              # Main Jupyter Notebook
├── shop_smart_ecommerce.csv      # Dataset (12,330 sessions)
├── assignment4_solution.py       # Full Python script
├── ShopSmart_ML_Report.pdf       # Client-ready report
│
├── outputs/
│   ├── eda_plots.png             # EDA visualisations
│   ├── confusion_matrices.png    # Baseline vs Pruned comparison
│   ├── decision_tree.png         # Pruned tree visualisation
│   └── feature_importance.png   # Feature importance chart
│
└── README.md
```

---

## 📊 Dataset Description

| Feature | Type | Description |
|---|---|---|
| Administrative / Duration | Numerical | Admin pages visited + time spent |
| Informational / Duration | Numerical | Info pages visited + time spent |
| ProductRelated / Duration | Numerical | Product pages visited + time spent |
| BounceRates | Numerical | Average bounce rate of visited pages |
| ExitRates | Numerical | Average exit rate of visited pages |
| **PageValues** | Numerical | ⭐ Strongest predictor of purchase |
| SpecialDay | Numerical | Closeness to a special day (0–1) |
| Month | Categorical | Month of the session (Feb–Dec) |
| VisitorType | Categorical | New / Returning / Other |
| Weekend | Boolean | True if session was on a weekend |
| **Revenue** | Boolean | 🎯 Target — True = Purchase made |

**Class Imbalance:** 84.5% No Purchase vs 15.5% Purchase → F1 Score used for evaluation

---

## ⚙️ Feature Preprocessing

```python
# Label encode categorical columns
le = LabelEncoder()
for col in ['Month', 'VisitorType']:
    df_processed[col] = le.fit_transform(df_processed[col])

# Convert boolean columns to int (0/1)
df_processed[['Weekend', 'Revenue']] = df_processed[['Weekend', 'Revenue']].astype(int)

# Train-Test Split — 80/20, stratified
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

> No feature scaling needed — Decision Trees are scale-invariant.

---

## 🌳 Model Training

### Baseline (Unpruned)
```python
dt_base = DecisionTreeClassifier(random_state=42)
dt_base.fit(X_train, y_train)
```

### Pruned (GridSearchCV)
```python
param_grid = {
    'max_depth': [3, 5, 7, 10, 15, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 5, 10],
    'ccp_alpha': [0.0, 0.001, 0.005, 0.01, 0.02]
}

grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid, scoring='f1', cv=5, n_jobs=-1
)
grid_search.fit(X_train, y_train)
```

---

## 📈 Results

| | Baseline (Unpruned) | Pruned (Optimised) | Benchmark |
|---|---|---|---|
| Tree Depth | 26 | 1 | — |
| Leaves | 810 | 2 | — |
| **F1 Score** | 0.5364 ❌ | **0.6432 ✅** | 0.55 |
| Precision (Purchase) | 0.52 | 0.56 | — |
| Recall (Purchase) | 0.55 | 0.76 | — |
| Accuracy | 0.85 | 0.87 | — |

**Best Parameters:**
```
ccp_alpha = 0.02  |  max_depth = 3  |  min_samples_leaf = 1  |  min_samples_split = 2
```

> 💡 Key insight: The pruned tree uses **only PageValues** to split — proving that a simpler model can outperform a complex one on imbalanced data.

---

## 📉 EDA Highlights

- **PageValues** is the strongest differentiator between buyers and non-buyers
- **November** has the highest purchase conversion rate across all months
- **New Visitors** convert at a higher rate than Returning Visitors
- Low **BounceRates** and **ExitRates** strongly correlate with purchases
- No missing values found in any column

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3 | Core language |
| Pandas | Data loading & manipulation |
| NumPy | Numerical operations |
| Seaborn / Matplotlib | EDA visualisations |
| Scikit-learn | ML model, GridSearchCV, metrics |
| ReportLab | Client PDF report generation |

---

## 🚀 How to Run

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/shopsmart-ml.git
cd shopsmart-ml

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn reportlab

# 3. Run the notebook
jupyter notebook _SHOPSMART.ipynb

# OR run the Python script directly
python assignment4_solution.py
```

---

## 🤖 Note on AI Usage

Parts of this project were built with assistance from **Claude (AI)** — particularly for refining plot formatting, understanding GridSearchCV parameters, and generating the client report.

AI was used as a learning tool — to elaborate on concepts, suggest improvements, and save time on boilerplate. All problem understanding, design decisions, and model logic were done independently.

---

## 📄 License

This project is for educational purposes (Apna College — Supervised ML Assignment 4).

---

*Made with 💙 while learning Machine Learning*
