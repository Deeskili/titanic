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


spotify_api = Blueprint('spotify_api', __name__, url_prefix='/api/spotify')
api = Api(spotify_api)
class Spotify_API(Resource):
    def __init__(self):
        # Read the Spotify dataset
        df = pd.read_csv('spotify.csv')

        # Initialize LabelEncoder
        self.label_encoder = LabelEncoder()
        
        # Encode the 'fav_song' column
        df['fav_song_encoded'] = self.label_encoder.fit_transform(df['fav_song'])

        # Split the data into features and target
        X = df.drop(columns=['fav_song', 'fav_song_encoded'])
        y = df['fav_song_encoded']

        # Split data into train and test sets
        self.X_train, _, self.y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create and train the linear regression model
        self.model = LinearRegression()
        self.model.fit(self.X_train, self.y_train)

    def predict_song(self, pop_song, country_song, classic_song, rap_song):
        # Make prediction
        prediction = self.model.predict([[pop_song, country_song, classic_song, rap_song]])
        return prediction.astype(int)[0]

    def post(self):
        try:
            data = request.json

            pop_song = data["pop_song"]
            country_song = data["country_song"]
            classic_song = data["classic_song"]
            rap_song = data["rap_song"]

            # Inverse transform the predicted label
            predicted_song_encoded = self.predict_song(pop_song, country_song, classic_song, rap_song)
            predicted_song = self.label_encoder.inverse_transform([predicted_song_encoded])[0]
            
            songs = [
                {'name': 'FEIN', 'description': 'Travis Scott_'},
                {'name': 'Creepin', 'description': 'The Weeknd'},
                {'name': 'Role Model', 'description': ' Brent Fiaiyaz'},
                {'name': 'Right my Wrongs', 'description': ' Bryson Tiller'},
                {'name': 'Roar', 'description': ' Katy Perry'},
                {'name': 'Shake it Off', 'description': ' Taylor Swift'},
                {'name': 'Often', 'description': ' The Weeknd_'},
                {'name': 'Thank God', 'description': ' Travis Scott'},
                {'name': 'Badtameez Dil', 'description': ' Yeh Jawaani Hai Deewani'},
                {'name': 'Same Old Love', 'description': ' Selena Gomez'},
                {'name': 'Waka Waka ', 'description': ' Shakira'},
                {'name': 'Swim', 'description': ' Chase Atlantic '},
                {'name': 'Fireside', 'description': ' Arctic Monkeys'},
                {'name': 'Daddy Issues', 'description': ' The Neighborhood'},
                {'name': 'Art Deco', 'description': 'Lana Del Rey '},
                {'name': 'The Color Violet', 'description': ' Tory Lanez '},
                {'name': 'Light it Up', 'description': ' Major Lazer'},
                {'name': 'Maneater', 'description': ' Nelly Furtado '},
                {'name': 'Dark Horse', 'description': ' Katy Perry '},
                {'name': 'Watch', 'description': ' Billie Eilish '},
                {'name': 'Eyes Without Face', 'description': ' Billy Idol '},
                {'name': 'Good Days', 'description': ' SZA'},
                {'name': 'sdp interlude', 'description': 'Travis Scott '},
                {'name': 'Mary', 'description': ' Alex G '},
                {'name': 'Flashing Lights', 'description': 'Kanye West '},
                {'name': 'Trust issues', 'description': 'Drake'},
                {'name': 'Gods Plan', 'description': 'Drake '},
                {'name': 'Time to Pretend', 'description': 'MGMT '},
                {'name': 'Sensei Wu', 'description': ' Ian Wu Father '}
            ]

            predicted_song = songs[predicted_song]
            return jsonify({'predicted_song': predicted_song})
        except Exception as e:
            return jsonify({'error': str(e)})

api.add_resource(Spotify_API, '/predict')
