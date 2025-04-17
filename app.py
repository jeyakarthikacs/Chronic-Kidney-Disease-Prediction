
# from flask import Flask, render_template, request
# import numpy as np
# import joblib

# app = Flask(__name__)
# model = joblib.load("xbg_model.pkl")

# @app.route("/")
# def home():
#     return render_template("index.html")

# @app.route("/predict", methods=["POST"])
# def predict():
#     # 1. Read form and cast
#     age    = float(request.form["age"])
#     sc     = float(request.form["sc"])
#     sg     = float(request.form["sg"])
#     htn    = int(request.form["htn"])
#     hemo   = float(request.form["hemo"])
#     dm     = int(request.form["dm"])
#     al     = float(request.form["al"])
#     appet  = int(request.form["appet"])
#     rc     = float(request.form["rc"])
#     pc     = int(request.form["pc"])

#     # 2. Build feature array
#     X_in = np.array([[age, sc, sg, htn, hemo, dm, al, appet, rc, pc]])

#     # 3. Predict
#     pred  = model.predict(X_in)[0]
#     proba = model.predict_proba(X_in)[0, pred]

#     # 4. Interpret
#     if pred == 0:
#         message = "🎉<br>You DON'T have Chronic Kidney Disease."
#         severity, stage, affected = "Normal", "Healthy", "0%"
#         img_url = "https://cdn-icons-png.flaticon.com/512/4319/4319163.png"
#     else:
#         message = "Oops! 🙁<br><br>You have CHRONIC KIDNEY DISEASE.<br><br>Please consult a doctor."
#         egfr = 186 * (sc ** -1.154) * (age ** -0.203)
#         # Map eGFR to CKD stages 1–5
#         if egfr >= 90:
#             severity = "Mild (Stage 1)"
#             stage    = "Stage 1"
#             affected_percentage = "15%"
#         elif egfr >= 60:
#             severity = "Mild (Stage 2)"
#             stage    = "Stage 2"
#             affected_percentage = "30%"
#         elif egfr >= 45:
#             severity = "Moderate (Stage 3)"
#             stage    = "Stage 3"
#             affected_percentage = "60%"
#         elif egfr >= 30:
#             severity = "Severe (Stage 4)"
#             stage    = "Stage 4"
#             affected_percentage = "85%"
#         else:
#             severity = "Kidney Failure (Stage 5)"
#             stage    = "Stage 5"
#             affected_percentage = "100%"
#         img_url = "https://cdni.iconscout.com/illustration/premium/thumb/male-doctor-and-female-doctor-4093612-3392962.png"

    # else:
    #     message = "Oops! 🙁<br>You have CHRONIC KIDNEY DISEASE.<br>Please consult a doctor."
    #     egfr = 186 * (sc ** -1.154) * (age ** -0.203)
    #     if egfr >= 90:
    #         severity, stage, affected = "Normal", "Healthy", "0%"
    #     elif egfr >= 60:
    #         severity, stage, affected = "Mild (Stage 1)", "Stage 1", "15%"
    #     elif egfr >= 45:
    #         severity, stage, affected = "Mild (Stage 2)", "Stage 2", "30%"
    #     elif egfr >= 30:
    #         severity, stage, affected = "Moderate (Stage 3)", "Stage 3", "60%"
    #     elif egfr >= 15:
    #         severity, stage, affected = "Severe (Stage 4)", "Stage 4", "85%"
    #     else:
    #         severity, stage, affected = "Kidney Failure (Stage 5)", "Stage 5", "100%"
    #     img_url = "https://cdni.iconscout.com/illustration/premium/thumb/male-doctor-and-female-doctor-4093612-3392962.png"

#     return render_template(
#         "result.html",
#         message=message,
#         severity=severity,
#         affected_percentage=affected,
#         stage=stage,
#         proba=f"{proba*100:.1f}%",
#         image_path=img_url,
#     )

# if __name__ == "__main__":
#     app.run(debug=True)
from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)
model = joblib.load("best_model_random_forest.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # 1) Read & cast form inputs
    age    = float(request.form["age"])
    sc     = float(request.form["sc"])
    sg     = float(request.form["sg"])
    htn    = int(request.form["htn"])
    hemo   = float(request.form["hemo"])
    dm     = int(request.form["dm"])
    al     = float(request.form["al"])
    appet  = int(request.form["appet"])
    rc     = float(request.form["rc"])
    pc     = int(request.form["pc"])

    # 2) Build feature array
    X_in = np.array([[age, sc, sg, htn, hemo, dm, al, appet, rc, pc]])

    # 3) Run model
    pred  = model.predict(X_in)[0]
    proba = model.predict_proba(X_in)[0, pred]

    # 4) Prepare defaults
    message = ""
    stage   = ""
    severity= ""
    affected_percentage = ""
    image_path = ""

    # 5) Fill in according to prediction
    if pred == 0:
        # Model says "no CKD"
        message = "🎉<br>You DON'T have Chronic Kidney Disease."
        stage   = "Healthy"
        severity= "Normal"
        affected_percentage = "0%"
        image_path = "https://cdn-icons-png.flaticon.com/512/4319/4319163.png"

    else:
        # Model says "CKD" → now compute eGFR to pick Stage 1–5
        message = "Oops! 🙁<br>You have CHRONIC KIDNEY DISEASE.<br>Please consult a doctor."
        egfr = 186 * (sc ** -1.154) * (age ** -0.203)

        if egfr >= 90:
            stage = "Stage 1"
            severity = "Mild (Stage 1)"
            affected_percentage = "15%"
        elif egfr >= 60:
            stage = "Stage 2"
            severity = "Mild (Stage 2)"
            affected_percentage = "30%"
        elif egfr >= 45:
            stage = "Stage 3"
            severity = "Moderate (Stage 3)"
            affected_percentage = "60%"
        elif egfr >= 30:
            stage = "Stage 4"
            severity = "Severe (Stage 4)"
            affected_percentage = "85%"
        else:
            stage = "Stage 5"
            severity = "Kidney Failure (Stage 5)"
            affected_percentage = "100%"

        image_path = "https://cdni.iconscout.com/illustration/premium/thumb/male-doctor-and-female-doctor-4093612-3392962.png"

    # 6) Render with a **consistent** set of variables
    return render_template(
        "result.html",
        message=message,
        stage=stage,
        severity=severity,
        affected_percentage=affected_percentage,
        proba=f"{proba*100:.1f}%",
        image_path=image_path
    )

if __name__ == "__main__":
    app.run(debug=True)


# from flask import Flask, render_template, request
# import numpy as np
# import joblib

# # Load the trained model
# model = joblib.load('xbg_model.pkl')

# app = Flask(__name__)

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if request.method == 'POST':
#         # Extract features from form
#         age = float(request.form.get('age', 0))
#         sc = float(request.form.get('sc', 1))  # serum creatinine
#         sg = float(request.form['sg'])         # specific gravity
#         htn = float(request.form['htn'])       # hypertension
#         hemo = float(request.form['hemo'])     # hemoglobin
#         dm = float(request.form['dm'])         # diabetes mellitus
#         al = float(request.form['al'])         # albumin
#         appet = float(request.form['appet'])   # appetite
#         rc = float(request.form['rc'])         # red blood cell count
#         pc = float(request.form['pc'])         # pus cell

#         # Input for the model
#         input_data = np.array([[sg, htn, hemo, dm, al, appet, rc, pc]])
#         prediction = model.predict(input_data)[0]

#         # eGFR Calculation using MDRD formula
#         eGFR = 186 * (sc ** -1.154) * (age ** -0.203)

#         if prediction == 0:
#             severity = "No CKD"
#             damage_percentage = "0%"
#         else:
#             if eGFR >= 90:
#                 severity = "Mild (Stage 1)"
#                 damage_percentage = "0% - 15%"
#             elif 60 <= eGFR < 90:
#                 severity = "Mild-Moderate (Stage 2)"
#                 damage_percentage = "16% - 30%"
#             elif 30 <= eGFR < 60:
#                 severity = "Moderate (Stage 3)"
#                 damage_percentage = "31% - 60%"
#             elif 15 <= eGFR < 30:
#                 severity = "Severe (Stage 4)"
#                 damage_percentage = "61% - 85%"
#             else:
#                 severity = "Critical (Stage 5 - Kidney Failure)"
#                 damage_percentage = "86% - 100%"

#         return render_template('result.html',
#                                prediction=prediction,
#                                severity=severity,
#                                ckd_percentage=damage_percentage)

# if __name__ == "__main__":
#     app.run(debug=True)

# Idhu last ah pannathu but default ah no nu kaamikithu
# from flask import Flask, render_template, request
# import numpy as np
# import joblib

# app = Flask(__name__)

# # Load the trained model (must be trained with 10 features)
# model = joblib.load("xbg_model.pkl")

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     # Get form inputs
#     age = float(request.form['age'])
#     sc = float(request.form['sc'])
#     sg = float(request.form['sg'])
#     htn = int(request.form['htn'])
#     hemo = float(request.form['hemo'])
#     dm = int(request.form['dm'])
#     al = float(request.form['al'])
#     appet = int(request.form['appet'])
#     rc = float(request.form['rc'])
#     pc = int(request.form['pc'])

#     # Prepare input
#     input_features = np.array([[age, sc, sg, htn, hemo, dm, al, appet, rc, pc]])
#     prediction = model.predict(input_features)

#     # Set result messages
#     if prediction[0] == 0:
#         message = "You do not have Chronic Kidney Disease."
#         severity = "Normal"
#         affected_percentage = "0%"
#         stage = "Healthy"
#         image_path = "https://cdn-icons-png.flaticon.com/512/4319/4319163.png"
#     else:
#         message = "You have CHRONIC KIDNEY DISEASE.\nPlease Consult Doctor."

#         # Calculate approximate eGFR
#         egfr = 186 * (sc ** -1.154) * (age ** -0.203)

#         if egfr >= 90:
#             severity = "Normal"
#             stage = "Healthy"
#             affected_percentage = "0%"
#         elif 60 <= egfr < 90:
#             severity = "Mild (Stage 1)"
#             stage = "Stage 1"
#             affected_percentage = "15%"
#         elif 45 <= egfr < 60:
#             severity = "Mild (Stage 2)"
#             stage = "Stage 2"
#             affected_percentage = "30%"
#         elif 30 <= egfr < 45:
#             severity = "Moderate (Stage 3)"
#             stage = "Stage 3"
#             affected_percentage = "60%"
#         elif 15 <= egfr < 30:
#             severity = "Severe (Stage 4)"
#             stage = "Stage 4"
#             affected_percentage = "85%"
#         else:
#             severity = "Kidney Failure (Stage 5)"
#             stage = "Stage 5"
#             affected_percentage = "100%"

#         image_path = "https://cdni.iconscout.com/illustration/premium/thumb/male-doctor-and-female-doctor-4093612-3392962.png"

#     return render_template('result.html',
#                            prediction=int(prediction[0]),
#                            message=message,
#                            severity=severity,
#                            affected_percentage=affected_percentage,
#                            stage=stage,
#                            image_path=image_path
#                            )


# if __name__ == '__main__':
#     app.run(debug=True)




# from flask import Flask, render_template, request
# import numpy as np
# import joblib

# app = Flask(__name__)

# # Load the model (make sure the filename matches)
# model = joblib.load("rf_model.pkl")  # Renamed from xbg_model.pkl

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Get inputs from form
#         age = float(request.form['age'])
#         sc = float(request.form['sc'])
#         sg = float(request.form['sg'])
#         htn = int(request.form['htn'])
#         hemo = float(request.form['hemo'])
#         dm = int(request.form['dm'])
#         al = float(request.form['al'])
#         appet = int(request.form['appet'])
#         rc = float(request.form['rc'])
#         pc = int(request.form['pc'])

#         # Prepare the input for prediction
#         input_features = np.array([[age, sc, sg, htn, hemo, dm, al, appet, rc, pc]])
#         prediction = model.predict(input_features)

#         # Determine the result
#         if prediction[0] == 0:
#             message = "🎉<br><br>You DON'T have Chronic Kidney Disease."
#             severity = "Normal"
#             affected_percentage = "0%"
#             stage = "Healthy"
#             image_path = "https://cdn-icons-png.flaticon.com/512/4319/4319163.png"
#         else:
#             message = "Oops! 🙁<br><br>You have CHRONIC KIDNEY DISEASE.<br><br>Please Consult Doctor."
#             egfr = 186 * (sc ** -1.154) * (age ** -0.203)
#             if egfr >= 90:
#                 severity = "Normal"
#                 stage = "Healthy"
#                 affected_percentage = "0%"
#             elif 60 <= egfr < 90:
#                 severity = "Mild (Stage 1)"
#                 stage = "Stage 1"
#                 affected_percentage = "15%"
#             elif 45 <= egfr < 60:
#                 severity = "Mild (Stage 2)"
#                 stage = "Stage 2"
#                 affected_percentage = "30%"
#             elif 30 <= egfr < 45:
#                 severity = "Moderate (Stage 3)"
#                 stage = "Stage 3"
#                 affected_percentage = "60%"
#             elif 15 <= egfr < 30:
#                 severity = "Severe (Stage 4)"
#                 stage = "Stage 4"
#                 affected_percentage = "85%"
#             else:
#                 severity = "Kidney Failure (Stage 5)"
#                 stage = "Stage 5"
#                 affected_percentage = "100%"

#             image_path = "https://cdni.iconscout.com/illustration/premium/thumb/male-doctor-and-female-doctor-4093612-3392962.png"

#         return render_template('result.html',
#                                prediction=int(prediction[0]),
#                                message=message,
#                                severity=severity,
#                                affected_percentage=affected_percentage,
#                                stage=stage,
#                                image_path=image_path)
#     except Exception as e:
#         return f"An error occurred: {e}"

# if __name__ == '__main__':
#     app.run(debug=True)
