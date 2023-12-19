from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer

app = Flask(__name__)

# Load your dataset
appdf = pd.read_csv(r"C:\ProjectFrontEnd\appdata_mergefinal11.csv", nrows=10000)

# Define features (X) and target variable (y)
X = appdf[['NAME_CONTRACT_TYPE_y', 'NAME_CLIENT_TYPE', 'CODE_GENDER', 'NAME_EDUCATION_TYPE',
        'NAME_INCOME_TYPE', 'NAME_FAMILY_STATUS', 'NAME_PORTFOLIO', 'OCCUPATION_TYPE',
        'NAME_GOODS_CATEGORY', 'PRODUCT_COMBINATION', 'NAME_PAYMENT_TYPE', 'CHANNEL_TYPE']]

y = appdf['NAME_CONTRACT_STATUS']

# Convert string labels to numerical values
label_encoder = LabelEncoder()
y_numeric = label_encoder.fit_transform(y)

# Identify categorical columns
categorical_cols = ['NAME_CONTRACT_TYPE_y', 'NAME_CLIENT_TYPE', 'CODE_GENDER', 'NAME_EDUCATION_TYPE',
                    'NAME_INCOME_TYPE', 'NAME_FAMILY_STATUS', 'NAME_PORTFOLIO', 'OCCUPATION_TYPE',
                    'NAME_GOODS_CATEGORY', 'PRODUCT_COMBINATION', 'NAME_PAYMENT_TYPE', 'CHANNEL_TYPE']

# Identify non-categorical columns
non_categorical_cols = X.columns.difference(categorical_cols)

# Create a random forest classifier with class weights
randF = RandomForestClassifier(random_state=42, class_weight='balanced')

# Create a pipeline with simplified preprocessing for RandomForestClassifier
pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
    ('classifier', randF)
])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_numeric, test_size=0.2, random_state=42)

# Train the classifier on the training data
pipeline.fit(X_train, y_train)

@app.route('/')
def home():
    return render_template('index1.html')

   

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get input data from the form
        input_data = {
            'NAME_CONTRACT_TYPE_y': [request.form['contract_type']],
            'NAME_CLIENT_TYPE': [request.form['client_type']],
            'CODE_GENDER': [request.form['gender']],
            'NAME_EDUCATION_TYPE': [request.form['education_type']],
            'NAME_INCOME_TYPE': [request.form['income_type']],
            'NAME_FAMILY_STATUS': [request.form['family_status']],
            'NAME_PORTFOLIO': [request.form['portfolio']],
            'OCCUPATION_TYPE': [request.form['occupation_type']],
            'NAME_GOODS_CATEGORY': [request.form['goods_category']],
            'PRODUCT_COMBINATION': [request.form['product_combination']],
            'NAME_PAYMENT_TYPE': [request.form['payment_type']],
            'CHANNEL_TYPE': [request.form['channel_type']]
        }

        # Preprocess the input data
        input_df = pd.DataFrame(input_data)
        print(input_df)
        # Apply the preprocessing pipeline
        input_preprocessed = pipeline.named_steps['encoder'].transform(pipeline.named_steps['imputer'].transform(input_df))
        print(input_preprocessed)
        # Make predictions
        prediction = pipeline.predict(input_preprocessed)

        # Map prediction to a numerical value
        prediction_value = int(prediction[0])

        prediction_probabilities = pipeline.predict_proba(input_preprocessed)

        # Map numerical prediction to labels
        prediction_label = 'Approved' if prediction_value == 1 else 'Rejected'
        


        result = {
            'prediction': prediction_label,
            'prediction_probabilities': prediction_probabilities.tolist(),
            'input_data': input_data
        }

        # Print additional information for debugging
       
        print(f"Prediction: {prediction_label}")
        return render_template('result.html', prediction_result=prediction_label)
         
      
        
if __name__ == '__main__':
    app.run(debug=True)