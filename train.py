import pickle
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
with open("dataset.pkl", "rb") as f:
    X, y = pickle.load(f)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Perceptron
model = Perceptron(max_iter=1000, tol=1e-3)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"✅ Model trained with accuracy: {accuracy * 100:.2f}%")

# Save the trained model
with open("perceptron_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model saved as perceptron_model.pkl")
