# Gradio app (for Hugging Face Spaces: rename to app.py, set sdk: gradio)
import joblib, pandas as pd, gradio as gr
model = joblib.load("model.pkl")

def predict(gender, married, dep, edu, self_emp, area, inc, coinc, amt, term, credit):
    row = pd.DataFrame([{"Gender":gender,"Married":married,"Dependents":dep,"Education":edu,
        "Self_Employed":self_emp,"Property_Area":area,"ApplicantIncome":inc,
        "CoapplicantIncome":coinc,"LoanAmount":amt,"Loan_Amount_Term":term,"Credit_History":int(credit)}])
    p = model.predict_proba(row)[0][list(model.classes_).index("Y")]
    return {"Approved": float(p), "Rejected": float(1-p)}

gr.Interface(predict, [
    gr.Dropdown(["Male","Female"], value="Male", label="Gender"),
    gr.Dropdown(["Yes","No"], value="Yes", label="Married"),
    gr.Dropdown(["0","1","2","3+"], value="0", label="Dependents"),
    gr.Dropdown(["Graduate","Not Graduate"], value="Graduate", label="Education"),
    gr.Dropdown(["No","Yes"], value="No", label="Self Employed"),
    gr.Dropdown(["Urban","Semiurban","Rural"], value="Urban", label="Property Area"),
    gr.Number(5000, label="Applicant Income"), gr.Number(0, label="Coapplicant Income"),
    gr.Number(130, label="Loan Amount (thousands)"),
    gr.Dropdown([360,480,300,240,180,120], value=360, label="Loan Term (months)"),
    gr.Radio(["1","0"], value="1", label="Credit History"),
], gr.Label(), title="Loan Approval Prediction").launch()
