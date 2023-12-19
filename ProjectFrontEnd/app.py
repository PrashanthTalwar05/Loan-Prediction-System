from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score

app = Flask(__name__)

appDf = pd.read_csv(r"C:\Users\PRASHANTH\Desktop\archive (3)\application_data2.csv")

# Assuming you have already created and updated the 'STATUS' column in appDf

# Select relevant columns for training
columns_for_training = ['AMT_INCOME_TOTAL', 'NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_REALTY', 'CNT_CHILDREN'] 
X = appDf[columns_for_training]

# Target variable
y = appDf['NAME_CONTRACT_STATUS']

# Split the data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Use one-hot encoding for categorical variables
categorical_cols = ['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_REALTY']
numeric_cols = ['AMT_INCOME_TOTAL', 'CNT_CHILDREN']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_cols),
        ('cat', OneHotEncoder(), categorical_cols)
    ])

X_train_preprocessed = preprocessor.fit_transform(X_train)

# Create a decision tree model
decision_tree = DecisionTreeClassifier()

# Train the model on the training set
decision_tree.fit(X_train_preprocessed, y_train)

# Define the preprocessor for new data
def preprocess_data(data):
    data = pd.DataFrame(data, index=[0])  # Convert to DataFrame with a single row
    data_preprocessed = preprocessor.transform(data)
    return data_preprocessed

# Define the route for the home page
@app.route('/')
def home():
    return render_template('index.html')  # Create an HTML file for the input form

# Define the route for predicting loan approval
# Define the route for predicting loan approval
# Define the route for predicting loan approval
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get input data from the form
        input_data = request.form.to_dict()

        # Preprocess the input data
        input_preprocessed = preprocess_data(input_data)

        # Make predictions
        prediction = decision_tree.predict(input_preprocessed)

        # Interpret the prediction and provide a simplified message
        result_message = 'There is a high Chances your Loan request will be Accepted' if prediction[0] == 1 else 'There is a high Chances your Loan request will be Rejected'

        # Return the result as a JSON object
        result = {'message': result_message}
        return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)


