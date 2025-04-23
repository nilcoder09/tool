import joblib
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# Load models
churn_model = joblib.load('./models/logistic_model.pkl')
spend_model = joblib.load('./models/linear_model.pkl')
rf_model = joblib.load('./models/rf_model.pkl')
kmeans_model = joblib.load('./models/kmeans_model.pkl')
scaler = joblib.load('./models/scaler.pkl')

@app.route("/", methods=["GET", "POST"])
def index():
    result = {}
    submitted = False

    if request.method == "POST":
        try:
            # Fetch form data
            sub_name = request.form.get("sub_name", "")
            cost = float(request.form.get("cost", 0))
            usage = int(request.form.get("usage", 0))
            satisfaction = int(request.form.get("satisfaction", 0))
            length = int(request.form.get("length", 0))
            payment = request.form.get("payment", "Other")
            region = request.form.get("region", "West")

            # Encoding categorical features
            payment_mapping = {"Credit Card": 0, "PayPal": 1, "Other": 2}
            region_mapping = {"North": 0, "South": 1, "East": 2, "West": 3}

            encoded_payment = payment_mapping.get(payment, 2)
            encoded_region = region_mapping.get(region, 3)

            # Prepare features
            features = np.array([[cost, usage, satisfaction, encoded_payment, encoded_region]])
            features_scaled = scaler.transform(features)

            # Predictions
            churn_pred = churn_model.predict(features_scaled)[0]
            spend_pred = spend_model.predict(features_scaled)[0]
            cluster = kmeans_model.predict(features_scaled)[0]

            cluster_labels = {
                0: "low spenders with high loyalty",
                1: "high spenders with moderate risk",
                2: "new users likely to explore",
                3: "moderate spenders with churn risk"
            }

            result = {
                "sub_name": sub_name,
                "churn": "high chance of churn" if churn_pred == 1 else "likely to stay subscribed",
                "predicted_spend": f"${round(spend_pred, 2)}",
                "cluster": f"Group {cluster} - {cluster_labels.get(cluster, 'undefined group')}"
            }

            submitted = True

        except Exception as e:
            print("Error:", e)

    return render_template("index.html", result=result, submitted=submitted)


@app.route("/report")
def report():
    return render_template("report.html")



if __name__ == "__main__":
    app.run(debug=True)
