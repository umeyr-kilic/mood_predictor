import pandas as pd
import requests
import json
import openpyxl
import datetime as dt
from scipy.stats import mode
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import lightgbm as lgb
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import chart_studio.plotly as py
import chart_studio
from PIL import Image
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
import joblib


def find_most_common(word_list):
    c = Counter(word_list)
    return c.most_common(1)[0][0]


def update_mood(predicted_mood):

    # Create figure
    fig = go.Figure()

    # Constants
    img_width = 1000
    img_height = 900
    scale_factor = 0.5

    # Add invisible scatter trace.
    # This trace is added to help the autoresize logic work.
    fig.add_trace(
        go.Scatter(
            x=[0, img_width * scale_factor],
            y=[0, img_height * scale_factor],
            mode="markers",
            marker_opacity=0
        )
    )

    # Configure axes
    fig.update_xaxes(
        visible=False,
        range=[0, img_width * scale_factor]
    )

    fig.update_yaxes(
        visible=False,
        range=[0, img_height * scale_factor],
        # the scaleanchor attribute ensures that the aspect ratio stays constant
        scaleanchor="x"
    )

    # Add image
    fig.add_layout_image(
        dict(
            x=0,
            sizex=img_width * scale_factor,
            y=img_height * scale_factor,
            sizey=img_height * scale_factor,
            xref="x",
            yref="y",
            opacity=1.0,
            layer="below",
            sizing="stretch",
            # source="https://raw.githubusercontent.com/michaelbabyn/plot_data/master/bridge.jpg")
            source=Image.open("{}.PNG".format(predicted_mood)))
    )

    # Configure other layout
    fig.update_layout(
        width=img_width * scale_factor,
        height=img_height * scale_factor,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
    )

    # Disable the autosize on double click because it adds unwanted margins around the image
    # More detail: https://plotly.com/python/configuration-options/
    fig.show(config={'doubleClick': 'reset'})

    chart_studio.tools.set_credentials_file(username='yumy', api_key='argNlrgZSJD8pmU4HVkn')
    url = py.plot(fig, filename='daily_mood')
    print(url)  # https://chart-studio.plotly.com/~yumy/46/#/


def predictor():
    # load
    loaded_rf = joblib.load("my_random_forest.joblib")

    predicted_mood = 'Calm'
    update_mood(predicted_mood)


def main():
    data = pd.read_excel('mood_status.xlsx', engine='openpyxl')

    data['Mood'] = pd.Categorical(data['Mood'])
    data['Mood_code'] = data['Mood'].cat.codes
    data['ETA'] = data['ETA'].fillna(0)

    # region extract features

    # avg of last 3 days of mood, usd
    data["mode_mood_3_days"] = data['Mood_code'].rolling(window=3).apply(lambda x: find_most_common(x)).fillna(0)
    data["avg_usd_3_days"] = data['USD/TRY'].transform(lambda x: x.rolling(3).mean()).fillna(0)

    # variation in last 7 days of mood, usd
    data["var_usd_3_days"] = data['USD/TRY'].rolling(3).var().fillna(0)
    data["mood_cc"] = pd.Categorical(data['Mood']).codes
    data["var_mood_3_days"] = data["mood_cc"].rolling(3).var().fillna(0)

    # usd increase rate of the day
    data["usd_previous_day"] = pd.Series([0]).append(data['USD/TRY'][:-1]).reset_index(drop=True)
    data["usd_increase_rate"] = (data['USD/TRY'] / data['usd_previous_day']).fillna(0).reset_index(drop=True)

    # day of week
    data['day_of_week'] = pd.to_datetime(data['Date']).dt.dayofweek
    # previous day usd, weather, mood
    data['usd_previous_day'] = pd.Series([0]).append(data['USD/TRY'][:-1]).reset_index(drop=True)
    data['weather_previous_day'] = pd.Series([0]).append(data['Weather'][:-1]).reset_index(drop=True)
    data['mood_previous_day'] = pd.Series([0]).append(data['Mood'][:-1]).reset_index(drop=True)

    # endregion
    labels = data['Mood']
    data.drop(columns=['Mood', 'Date'], inplace=True)

    # weather one hot conversion
    one_hot = pd.get_dummies(data['Weather'])
    data.drop(columns=['Weather'], inplace=True)
    data = data.join(one_hot)

    # weather_previous_day one hot conversion
    one_hot = pd.get_dummies(data['weather_previous_day'], prefix="previous_day", drop_first=True)
    data.drop(columns=['weather_previous_day'], inplace=True)
    data = data.join(one_hot)

    # mood_previous_day one hot conversion
    one_hot = pd.get_dummies(data['mood_previous_day'], prefix="previous_day", drop_first=True)
    data.drop(columns=['mood_previous_day'], inplace=True)
    data = data.join(one_hot)

    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.fillna(0, inplace=True)

    # region random forest classifier
    X, X_test, y, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    rf = RandomForestClassifier(max_depth=2, random_state=0, n_estimators=10)
    rf.fit(X, y)
    y_predict = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_predict)

    y_test = y_test.to_numpy()
    print("Random Forest Accuracy is: {}".format(accuracy))

    # endregion

    cm = confusion_matrix(y, rf.predict(X), labels=rf.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf.classes_)
    disp.plot()
    plt.title("Confusion Matrix")
    plt.show()

    # save model
    joblib.dump(rf, "random_forest.joblib")


if __name__ == '__main__':
    main()

