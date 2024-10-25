# Import libraries
import pandas as pd # type: ignore
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.linear_model import LinearRegression # type: ignore
import pickle

# Load the dataset
data = pd.read_csv('Admission_Predict.csv')

# Features and target variable
X = data[['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA', 'Research']]
y = data['Chance of Admit ']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model to a pickle file
pickle.dump(model, open('model.pkl', 'wb'))

print("Model training complete and saved to model.pkl")
y_pred=model.predict(X_test)
r2 = r2_score(y_test, y_pred)

# Convert R-squared to percentage
accuracy_percentage = r2 * 100

print(f"The accuracy of the model is {accuracy_percentage:.2f}%")
