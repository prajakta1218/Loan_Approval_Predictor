# Streamlit app: streamlit run app.py
import joblib, pandas as pd, streamlit as st

st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦")
st.title("🏦 Loan Approval Prediction")
st.caption("ISE-2 | Machine Learning (1ICPC406) | Random Forest model")

@st.cache_resource
def load(): return joblib.load("model.pkl")
model = load()

c1, c2 = st.columns(2)
gender = c1.selectbox("Gender", ["Male","Female"])
married = c2.selectbox("Married", ["Yes","No"])
dep = c1.selectbox("Dependents", ["0","1","2","3+"])
edu = c2.selectbox("Education", ["Graduate","Not Graduate"])
self_emp = c1.selectbox("Self Employed", ["No","Yes"])
area = c2.selectbox("Property Area", ["Urban","Semiurban","Rural"])
inc = c1.number_input("Applicant Income (monthly)", 0, value=5000, step=500)
coinc = c2.number_input("Coapplicant Income (monthly)", 0, value=0, step=500)
amt = c1.number_input("Loan Amount (in thousands)", 1, value=130)
term = c2.selectbox("Loan Term (months)", [360,480,300,240,180,120])
credit = st.radio("Credit History (1 = good)", [1,0], horizontal=True)

if st.button("Predict", type="primary"):
    row = pd.DataFrame([{"Gender":gender,"Married":married,"Dependents":dep,"Education":edu,
        "Self_Employed":self_emp,"Property_Area":area,"ApplicantIncome":inc,
        "CoapplicantIncome":coinc,"LoanAmount":amt,"Loan_Amount_Term":term,"Credit_History":credit}])
    pred = model.predict(row)[0]
    prob = model.predict_proba(row)[0][list(model.classes_).index("Y")]
    (st.success if pred == "Y" else st.error)(
        f"{'✅ Loan Approved' if pred=='Y' else '❌ Loan Not Approved'} (approval probability: {prob:.0%})")
