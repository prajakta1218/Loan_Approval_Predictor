import os, joblib, numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

CAT = ["Gender","Married","Dependents","Education","Self_Employed","Property_Area"]
NUM = ["ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term","Credit_History"]

def synthetic(n=1500, seed=42):
    r = np.random.default_rng(seed)
    df = pd.DataFrame({
        "Gender": r.choice(["Male","Female"], n, p=[.8,.2]),
        "Married": r.choice(["Yes","No"], n, p=[.65,.35]),
        "Dependents": r.choice(["0","1","2","3+"], n, p=[.55,.2,.15,.1]),
        "Education": r.choice(["Graduate","Not Graduate"], n, p=[.78,.22]),
        "Self_Employed": r.choice(["Yes","No"], n, p=[.14,.86]),
        "ApplicantIncome": r.lognormal(8.5, .5, n).round(),
        "CoapplicantIncome": (r.lognormal(7.5, .8, n) * r.integers(0, 2, n)).round(),
        "LoanAmount": r.normal(140, 60, n).clip(20, 500).round(),
        "Loan_Amount_Term": r.choice([120,180,240,300,360,480], n, p=[.02,.05,.03,.05,.82,.03]),
        "Credit_History": r.choice([1,0], n, p=[.84,.16]),
        "Property_Area": r.choice(["Urban","Semiurban","Rural"], n),
    })
    income = df.ApplicantIncome + df.CoapplicantIncome
    score = (3*df.Credit_History + 1.2*(income/(df.LoanAmount*1000/df.Loan_Amount_Term*12+1) > .6)
             + .4*(df.Property_Area=="Semiurban") + .3*(df.Married=="Yes")
             + .3*(df.Education=="Graduate") + r.normal(0, .6, n))
    df["Loan_Status"] = np.where(score > 2.2, "Y", "N")
    return df

df = pd.read_csv("loan.csv") if os.path.exists("loan.csv") else synthetic()
df = df.drop(columns=[c for c in ["Loan_ID"] if c in df.columns]).dropna(subset=["Loan_Status"])
X, y = df[CAT+NUM], df["Loan_Status"]
for c in CAT: X[c] = X[c].fillna(X[c].mode()[0]).astype(str)
for c in NUM: X[c] = X[c].fillna(X[c].median())

pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CAT)], remainder="passthrough")
model = Pipeline([("pre", pre), ("rf", RandomForestClassifier(300, max_depth=8, random_state=42))])
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
model.fit(Xtr, ytr)
p = model.predict(Xte)
print("Accuracy:", round(accuracy_score(yte, p), 3)); print(classification_report(yte, p))
joblib.dump(model, "model.pkl")
