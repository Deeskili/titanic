import threading

# import "packages" from flask
from flask import render_template,request  # import render_template from "public" flask libraries
from flask.cli import AppGroup


# import "packages" from "this" project
from __init__ import app, db, cors  # Definitions initialization

# setup App pages
from projects.projects import app_projects # Blueprint directory import projects definition

import threading
# import "packages" from flask
from flask import render_template,request  # import render_template from "public" flask libraries
from flask.cli import AppGroup
from flask import Flask
from flask_cors import CORS
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder


# import "packages" from "this" project
from __init__ import app, db, cors  # Definitions initialization

import json, jwt
from flask import Blueprint, request, jsonify, current_app, Response
from flask_restful import Api, Resource # used for REST API building
from datetime import datetime
from auth_middleware import token_required

from model.users import User
# import "packages" from "this" project
from __init__ import app, db, cors  # Definitions initialization

# setup APIs
from api.covid import covid_api # Blueprint import api definition
from api.joke import joke_api # Blueprint import api definition
from api.user import user_api # Blueprint import api definition
from api.player import player_api
# database migrations
from model.users import initUsers
from model.players import initPlayers
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# setup App pages
from projects.projects import app_projects # Blueprint directory import projects definition


iris_api = Blueprint('iris_api', __name__, url_prefix='/api/iris')
api = Api(iris_api)

class IrisAPI(Resource):
    def __init__(self):
        # Read the Iris dataset
        df = pd.read_csv('iris.csv')

        # Encode the 'variety' column
        self.label_encoder = LabelEncoder()
        df['variety'] = self.label_encoder.fit_transform(df['variety'])

        # Split the data into features and target
        X = df.drop(columns=['variety'])
        y = df['variety']

        # Split data into train and test sets
        self.X_train, _, self.y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create and train the linear regression model
        self.model = LinearRegression()
        self.model.fit(self.X_train, self.y_train)

    def predict_variety(self, sepal_length, sepal_width, petal_length, petal_width):
        # Make prediction
        prediction = self.model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
        # Inverse transform the encoded prediction to get original variety
        predicted_variety = self.label_encoder.inverse_transform(prediction.astype(int))
        return predicted_variety[0]

    def post(self):
        try:
            data = request.json

            sepal_length = data["sepal_length"]
            sepal_width = data["sepal_width"]
            petal_length = data["petal_length"]
            petal_width = data["petal_width"]

            predicted_variety = self.predict_variety(sepal_length, sepal_width, petal_length, petal_width)
            
            print(jsonify({'predicted_variety': predicted_variety}))
            return jsonify({'predicted_variety': predicted_variety})
        except Exception as e:
            return jsonify({'error': str(e)})

api.add_resource(IrisAPI, '/predict')
