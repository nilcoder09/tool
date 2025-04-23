import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Load dataset
df = pd.read_csv('../data/cleaned_data.csv')

# Fill missing numeric values just in case
df = df.fillna(df.mean(numeric_only=True))

# Define features
features = [
    'Monthly_Spend',             # cost
    'Support_Tickets_Raised',    # usage
    'Satisfaction_Score',        # satisfaction
    'Payment_Method',            # categorical
    'Region',                    # categorical
    'Subscription_Length'        # length
]

# Subset
X_raw = df[features].copy()

# Encode categorical features
payment_mapping = {"Credit Card": 0, "PayPal": 1, "Other": 2}
region_mapping = {"North": 0, "South": 1, "East": 2, "West": 3}

X_raw.loc[:, 'Payment_Method'] = X_raw['Payment_Method'].map(payment_mapping)
X_raw.loc[:, 'Region'] = X_raw['Region'].map(region_mapping)

# Drop rows with any remaining NaNs after mapping
X_raw = X_raw.dropna()

# Align labels (filtered rows)
y_churn = df.loc[X_raw.index, 'Churned']
y_spend = df.loc[X_raw.index, 'Monthly_Spend']

# Split
X_train, X_test, y_train, y_test = train_test_split(X_raw, y_churn, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_scaled, y_train)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

spend_model = LinearRegression()
spend_model.fit(X_train_scaled, y_spend.loc[X_train.index])

kmeans_model = KMeans(n_clusters=3, random_state=42)
kmeans_model.fit(X_train_scaled)

# Save models + scaler
os.makedirs('../outputs/models', exist_ok=True)
joblib.dump(log_model, '../outputs/models/logistic_model.pkl')
joblib.dump(rf_model, '../outputs/models/rf_model.pkl')
joblib.dump(spend_model, '../outputs/models/linear_model.pkl')
joblib.dump(kmeans_model, '../outputs/models/kmeans_model.pkl')
joblib.dump(scaler, '../outputs/models/scaler.pkl')

print("✅ Models trained and saved successfully!")




# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.cluster import KMeans
# from sklearn.metrics import classification_report, mean_squared_error
# import joblib
#
# # Load the data
# df = pd.read_csv('../data/cleaned_data.csv')
# # Impute missing values with the column's mean for numerical columns
# df.fillna(df.mean(), inplace=True)
#
# # Display the first few rows to check the data
# print(df.head())
# # Feature and target columns
# features = ['Subscription_Length', 'Satisfaction_Score', 'Discount_Offered',
#             'Support_Tickets_Raised', 'gender_encoded']  # These are the features
# X = df[features]  # Features (independent variables)
# y = df['Churned']  # Target (dependent variable)
#
# # Encoding categorical variables (if necessary)
# # Assuming 'gender_encoded' is already numeric and no other categorical features need encoding.
#
# # Split data into train and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Standardize the features (important for models like Logistic Regression and KMeans)
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
# # Train logistic regression model for churn prediction
# log_reg = LogisticRegression(max_iter=1000)
# log_reg.fit(X_train_scaled, y_train)
#
# # Make predictions
# y_pred_log = log_reg.predict(X_test_scaled)
#
# # Evaluate the model
# print("Logistic Regression Classification Report:")
# print(classification_report(y_test, y_pred_log))
# # Train linear regression model for predicted spend
# lin_reg = LinearRegression()
# lin_reg.fit(X_train_scaled, df['Monthly_Spend'][y_train.index])  # Use 'Monthly_Spend' as the target
#
# # Make predictions
# spend_pred = lin_reg.predict(X_test_scaled)
#
# # Evaluate the model
# print("Linear Regression Mean Squared Error:")
# print(mean_squared_error(df['Monthly_Spend'][y_test.index], spend_pred))
# # Train random forest model for churn prediction
# rf = RandomForestClassifier(n_estimators=100, random_state=42)
# rf.fit(X_train_scaled, y_train)
#
# # Make predictions
# y_pred_rf = rf.predict(X_test_scaled)
#
# # Evaluate the model
# print("Random Forest Classification Report:")
# print(classification_report(y_test, y_pred_rf))
# # Train KMeans model for clustering
# kmeans = KMeans(n_clusters=4, random_state=42)
# kmeans.fit(X_train_scaled)
#
# # Make predictions (cluster assignments)
# cluster_pred = kmeans.predict(X_test_scaled)
#
# # Display clusters
# print("KMeans Cluster Labels:")
# print(cluster_pred)
# # Save models
# joblib.dump(log_reg, '../models/logistic_model.pkl')
# joblib.dump(lin_reg, '../models/linear_model.pkl')
# joblib.dump(rf, '../models/rf_model.pkl')
# joblib.dump(kmeans, '../models/kmeans_model.pkl')
#
# # Save the scaler for feature scaling in the web app
# joblib.dump(scaler, '../models/scaler.pkl')



