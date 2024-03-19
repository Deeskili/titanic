import pandas as pd
import seaborn as sns
from flask import Flask, request, jsonify
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the titanic dataset
titanic_data = sns.load_dataset('titanic')
print("Titanic Data")
print(titanic_data[['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'alone']])

# Preprocess the data
td = titanic_data.copy()
td.drop(['alive', 'who', 'adult_male', 'class', 'embark_town', 'deck'], axis=1, inplace=True)
td.dropna(inplace=True)  # drop rows with at least one missing value, after dropping unuseful columns
td['sex'] = td['sex'].apply(lambda x: 1 if x == 'male' else 0)
td['alone'] = td['alone'].apply(lambda x: 1 if x else 0)

# Encode categorical variables
enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(td[['embarked']])
onehot = enc.transform(td[['embarked']]).toarray()
cols = ['embarked_' + val for val in enc.categories_[0]]
td[cols] = pd.DataFrame(onehot) 
td.drop(['embarked'], axis=1, inplace=True)
td.dropna(inplace=True)  # drop rows with at least one missing value, after preparing the data
print(td.columns)
print(td)

print(td.select_dtypes(include=['number']).median())
# Calculate the mean of numeric columns only
numeric_columns = titanic_data.select_dtypes(include=['number'])
mean_numeric_columns = numeric_columns.mean()
print(mean_numeric_columns)

print(td.query("survived == 1").mean())
print("maximums for survivors")
print(td.query("survived == 1").max())
print()
print("minimums for survivors")
print(td.query("survived == 1").min())

# Build distinct data frames on survived column
X = td.drop('survived', axis=1)  # all except 'survived'
y = td['survived']  # only 'survived'

# Split arrays in random train 70%, random test 30%, using stratified sampling (same proportion of survived in both sets) and a fixed random state (42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a decision tree classifier
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

# Test the model
y_pred = dt.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('DecisionTreeClassifier Accuracy: {:.2%}'.format(accuracy))

# Train a logistic regression model
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# Test the model
y_pred = logreg.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('LogisticRegression Accuracy: {:.2%}'.format(accuracy))

# Define the Flask app
app = Flask(__name__)

# Define the TitanicRegression class
class TitanicRegression:
    def __init__(self):
        self.encoder = None
        self.logreg = None

    def initTitanic(self):
        titanic_data = sns.load_dataset('titanic')
        X = titanic_data.drop('survived', axis=1)
        y = titanic_data['survived']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Initialize the encoder
        self.encoder = OneHotEncoder(handle_unknown='ignore')
        self.X_train = self.encoder.fit_transform(X_train)
        self.X_test = self.encoder.transform(X_test)

        # Train a logistic regression model
        self.logreg = LogisticRegression()
        self.logreg.fit(self.X_train, y_train)

    def predictSurvival(self, passenger):
        passenger_df = pd.DataFrame([passenger], columns=['name','pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'alone'])

        # Preprocess the passenger data
        passenger_encoded = self.encoder.transform(passenger_df)

        predict = self.logreg.predict(passenger_encoded)
        return predict


# Initialize the Titanic model
titanic_model = TitanicRegression()
titanic_model.initTitanic()

# Define the API endpoint for prediction
@app.route('/api/predict', methods=['POST'])
def predict():
    # Get the passenger data from the request
    json_data = request.get_json()


    name = json_data.get('name')
    pclass = json_data.get('pclass')
    sex = json_data.get('sex')
    age = json_data.get('age')
    sibsp = json_data.get('sibsp')
    parch = json_data.get('parch')
    fare = json_data.get('fare')
    embarked = json_data.get('embarked')
    alone = json_data.get('alone')

    if alone == "true":
        alone = True
    elif alone == "false":
        alone = False

    data_list = [name, pclass, sex, age, sibsp, parch, fare, embarked, alone]    

    # Use the model to predict survival
    response = titanic_model.predictSurvival(data_list)

    # Return the response as JSON
    return jsonify(response)

if __name__ == "__main__":
    # Run the Flask app
    app.run(debug=True, host="0.0.0.0", port="8086")
