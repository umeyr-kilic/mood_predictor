import pandas as pd
import requests
import json
import openpyxl
import datetime as dt
from scipy.stats import mode
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import chart_studio.plotly as py
import chart_studio
import plotly.io as pio
from PIL import Image


def find_most_common(word_list):
    c = Counter(word_list)
    return c.most_common(1)[0][0]


def predictor(classifier_type):
    pass


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


def main():
    data = pd.read_excel('mood_status.xlsx', engine='openpyxl')

    data['Mood'] = pd.Categorical(data['Mood'])
    data['Mood_code'] = data['Mood'].cat.codes
    data['ETA'] = data['ETA'].fillna(0)

    # region extract features

    # avg of last 3 days of mood, usd
    data["mode_mood_7_days"] = data['Mood_code'].rolling(window=3).apply(lambda x: find_most_common(x)).fillna(0)
    data["avg_usd_7_days"] = data['USD/TRY'].transform(lambda x: x.rolling(3).mean()).fillna(0)

    # variation in last 7 days of mood, usd
    # usd increase rate of the day
    # day of week
    # previous day usd, weather, mood

    # endregion

    labels = data['Mood']
    data.drop(columns=['Mood', 'Date'], inplace=True)

    one_hot = pd.get_dummies(data['Weather'])
    data.drop(columns=['Weather'], inplace=True)
    data = data.join(one_hot)

    # region classifier
    X, X_test, y, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    rf = RandomForestClassifier(max_depth=2, random_state=0, n_estimators=10)
    rf.fit(X, y)
    accuracy = accuracy_score(y_test, rf.predict(X_test))
    print("Classification Accuracy is: {}".format(accuracy))
    # endregion

    update_mood("calm")


if __name__ == '__main__':
    main()

