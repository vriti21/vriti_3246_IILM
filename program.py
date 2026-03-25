# DECISION TREE CLASSIFIER
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load dataset
data = pd.read_csv(r"C:\Users\Vriti Gupta\Downloads\climate_change_pest_control_impacts.csv")  # your dataset


# Step 2: Encode categorical columns
encoder = LabelEncoder()
data['crop_type'] = encoder.fit_transform(data['crop_type'])
data['pest_type'] = encoder.fit_transform(data['pest_type'])

# Step 3: Select features
X = data[['temperature', 'rainfall', 'humidity', 'crop_type']]
y = data['pest_risk']   # high / medium / low

# Step 4: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Step 5: Train Decision Tree Model
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

# Step 6: Predict
pred = dt.predict(X_test)

# Step 7: Evaluate
print("Decision Tree Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

# -----------------------------
# RANDOM FOREST CLASSIFIER
# -----------------------------
from sklearn.ensemble import RandomForestClassifier

# Reuse same X_train, X_test, y_train, y_test from previous code

rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

pred_rf = rf.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(y_test, pred_rf))
print(classification_report(y_test, pred_rf))

# -----------------------------
# ARTIFICIAL NEURAL NETWORK
# -----------------------------
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# Encode target labels to numbers (ANN needs numeric output)
y_encoded = encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.25, random_state=42)

# Build ANN model
model = Sequential([
    Dense(16, activation='relu', input_shape=(4,)),  # input layer
    Dense(12, activation='relu'),                    # hidden layer
    Dense(3, activation='softmax')                   # output layer (3 classes)
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train ANN
model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=1)

# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print("ANN Accuracy:", acc)

# Example: Predict pest risk for current climate conditions
sample = pd.DataFrame({
    'temperature': [32],
    'rainfall': [120],
    'humidity': [78],
    'crop_type': [1],  # encoded value
})

print("Decision Tree Prediction:", dt.predict(sample)[0])
print("Random Forest Prediction:", rf.predict(sample)[0])
print("ANN Prediction:", encoder.inverse_transform([model.predict(sample)[0].argmax()])[0])

def recommend(risk):
    if risk == 'high':
        return "Use eco-friendly biological control; avoid chemical overuse."
    elif risk == 'medium':
        return "Limited pesticide use + regular monitoring suggested."
    else:
        return "Low risk – no chemical control required."

print(recommend("high"))
