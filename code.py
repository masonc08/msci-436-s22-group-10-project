import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd


@st.cache(allow_output_mutation=True)
def load_model():
    df = pd.read_csv("genres_v2.csv", sep=r',', skipinitialspace=True)

    # Replacing NaN with default valuesb
    df = df.fillna(value=-1)

    # Dropping columns
    df = df.drop(['id', 'uri', 'track_href', 'analysis_url', 'time_signature', 'type', 'mode', 'key'], axis=1)
    df['genre'].replace(
        to_replace=['Dark Trap', 'dnb', 'Emo', 'hardstyle', 'Hiphop', 'Pop', 'psytrance', 'Rap', 'RnB', 'techhouse',
                    'techno', 'trance', 'trap', 'Trap Metal', 'Underground Rap'],
        value=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], inplace=True)
    df = df.astype({"song_name": str})

    # Columns used as predictors
    X = df.drop(["genre", "song_name"], axis=1)
    y = df["genre"]

    # Training Model
    X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, random_state=0, test_size=0.1)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    rfc = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                                max_depth=None, max_features='auto', max_leaf_nodes=None,
                                min_samples_leaf=1, min_samples_split=2,
                                min_weight_fraction_leaf=0.0, n_estimators=600, n_jobs=1,
                                oob_score=False, random_state=None, verbose=0,
                                warm_start=False)
    rfc.fit(X_train, y_train)
    return rfc

@st.cache
def run_model(input_params: dict):
    prediction = list(rfc.predict(input_params.values).astype(int)).pop()

    genre_dict = {
        1: 'Dark Trap',
        2: 'dnb',
        3: 'Emo',
        4: 'hardstyle',
        5: 'Hiphop',
        6: 'Pop',
        7: 'psytrance',
        8: 'Rap',
        9: 'RnB',
        10: 'techhouse',
        11: 'techno',
        12: 'trance',
        13: 'trap',
        14: 'Trap Metal',
        15: 'Underground Rap',
    }
    return genre_dict[prediction]

@st.cache
def recommendation(model_output: str):
    df = pd.read_csv("genres_v2.csv", sep=r',', skipinitialspace=True)
    df2 = df[df["genre"] == model_output]

    return(df2["song_name"].drop_duplicates())

st.title('Find out what genre to listen to!')

rfc = load_model()

danceability = st.slider('Preferred danceability (enter as a percent between 0-100', 0, 100)
st.write("You entered a danceability of ", danceability, '% ')

energy = st.slider('Preferred energy (enter as a percent between 0-100', 0, 100)
st.write("You entered a energy of ", energy, '% ')

speechiness = st.slider('Preferred speechiness (enter as a percent between 0-100', 0, 100)
st.write("You entered a speechiness of ", speechiness, '% ')

instrumentalness = st.slider('Preferred instrumentalness (enter as a percent between 0-100', 0, 100)
st.write("You entered a instrumentalness of ", instrumentalness, '% ')

liveness = st.slider('Preferred liveness (enter as a percent between 0-100', 0, 100)
st.write("You entered a liveness of ", liveness, '% ')

valence = st.slider('Preferred valence (enter as a percent between 0-100', 0, 100)
st.write("You entered a valence of ", valence, '% ')

acousticness = st.slider('Preferred acousticness (enter as a percent between 0-100', 0, 100)
st.write("You entered a acousticness of ", acousticness, '% ')

tempo = st.slider('Preferred tempo (ms) between 0-250', 0, 250)
st.write("You entered a tempo of ", tempo)

duration = st.slider('Preferred duration (in ms)' , 0, 1000000)
st.write("You entered a duration of ", duration)

loudness = st.slider('Preferred loudness', -50, 10)
st.write("You entered a loudness of ", loudness)

input_dict = {
    'danceability': [danceability/100],
    'energy': [energy/100],
    'loudness': [loudness],
    'speechiness': [speechiness/100],
    'acousticness': [acousticness/100],
    'instrumentalness': [instrumentalness/100],
    'liveness': [liveness/100],
    'valence': [valence/100],
    'tempo': [tempo],
    'duration_ms': [duration],
}

input_params = pd.DataFrame.from_dict(input_dict)


run = st.button('Run')

if run:
    model_output = run_model(input_params)
    st.title("You will enjoy the following music genre: " + str(model_output))

    recommendation_df = recommendation(model_output)
    st.title("Try these songs!: ")
    st.dataframe(recommendation_df)

