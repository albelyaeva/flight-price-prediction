import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("archive/flight_data.csv")
if "Unnamed: 0" in df.columns:
    df.drop(columns=["Unnamed: 0"], inplace=True)


time_map = {'Early Morning': 1, 'Morning': 2, 'Afternoon': 3, 'Evening': 4, 'Night': 5, 'Late Night': 6}
df['departure_time'] = df['departure_time'].map(time_map)
df['arrival_time'] = df['arrival_time'].map(time_map)

categorical_cols = df.select_dtypes(include='object').columns.tolist()
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    le.classes_ = np.append(le.classes_, "unknown")
    label_encoders[col] = le

df.fillna(0, inplace=True)

X = df.drop('price', axis=1)
y = df['price']

X_train_full, X_temp, y_train_full, y_temp = train_test_split(X, y, test_size=0.4, random_state=1)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=1)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_full)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=1, n_jobs=-1)
model.fit(X_train_full, y_train_full)


y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"Test MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.3f}")

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(label_encoders, open("label_encoders.pkl", "wb"))
pickle.dump(list(X.columns), open("features.pkl", "wb"))

print("Model and preprocessing saved.")
