from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

def train_decision_tree_model(X_train, y_train, X_test, y_test):
    # Use one-hot encoding for categorical variables
    categorical_cols = ['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_REALTY']
    numeric_cols = ['AMT_INCOME_TOTAL', 'CNT_CHILDREN']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_cols),
            ('cat', OneHotEncoder(), categorical_cols)
        ])

    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)

    # Create a decision tree model
    decision_tree = DecisionTreeClassifier()

    # Train the model on the training set
    decision_tree.fit(X_train_preprocessed, y_train)

    # Predictions on the training set
    y_train_pred = decision_tree.predict(X_train_preprocessed)

    # Calculate accuracy for the training set
    accuracy_train = accuracy_score(y_train, y_train_pred)
    print(f'Accuracy on the training set: {accuracy_train:.2f}')

    # Predictions on the test set
    y_pred = decision_tree.predict(X_test_preprocessed)

    # Calculate accuracy for the test set
    accuracy_test = accuracy_score(y_test, y_pred)
    print(f'Accuracy on the test set: {accuracy_test:.2f}')

    return decision_tree

if __name__ == '__main__':
    # Provide the necessary data for training and testing
    # For example, replace 'your_data.csv' with the actual dataset path
    df = pd.read_csv('your_data.csv')
    columns_for_training = ['AMT_INCOME_TOTAL', 'NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_REALTY', 'CNT_CHILDREN']
    X = df[columns_for_training]
    y = df['STATUS']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train and evaluate the decision tree model
    trained_model = train_decision_tree_model(X_train, y_train, X_test, y_test)

    # Save the trained model (optional)
    joblib.dump(trained_model, 'decision_tree_model.joblib')
