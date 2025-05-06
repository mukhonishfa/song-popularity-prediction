import streamlit as st
import streamlit.components.v1 as stc
import pickle

with open('Elastic_Net_Model.pkl', 'rb') as file:
    Elastic_Net_Model = pickle.load(file)

html_temp = """<div style="background-color:#000;padding:10px;border-radius:10px">
                <h1 style="color:#fff;text-align:center">Song Popularity Prediction App</h1> 
                <h4 style="color:#fff;text-align:center">Made for: Credit Team</h4> 
                """

desc_temp = """ ### Song Popularity Prediction App 
                This app is used by Cakrawala Team for predicting Song Popularity
                    
                #### Data Source
                Kaggle: https://www.kaggle.com/datasets/yasserh/song-popularity-dataset
                """

def main():
    stc.html(html_temp)
    menu = ["Home", "Machine Learning App"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Home")
        st.markdown(desc_temp, unsafe_allow_html=True)
    elif choice == "Machine Learning App":
        run_ml_app()

def run_ml_app():
    design = """<div style="padding:15px;">
                    <h1 style="color:#fff">Song Popularity Prediction</h1>
                </div
             """
    st.markdown(design, unsafe_allow_html=True)
    left,right = st.columns((2,2))
    song_duration_ms = left.number_input('Song Duration (ms)', min_value=100000, max_value=350000)
    acousticness = right.number_input('Acousticness', min_value=0.00, max_value=1.00)
    danceability = left.number_input('Danceability', min_value=0.00, max_value=1.00)
    energy = right.number_input('Energy', min_value=0.00, max_value=1.00)
    instrumentalness = left.number_input('Instrumentalness', min_value=0.00, max_value=1.00)
    key = right.number_input('Key', min_value=0, max_value=11)
    liveness = left.number_input('Liveness', min_value=0.00, max_value=1.00)
    loudness = right.number_input('Loudness', min_value=-35.00, max_value=1.60)
    audio_mode = left.selectbox('Audio Mode',(1,0))
    speechiness = right.number_input('Speechiness', min_value=0.00, max_value=1.00)
    tempo = left.number_input('Tempo', min_value=0.00, max_value=243.00)
    time_signature = right.number_input('Time Signature', min_value=0, max_value=5)
    audio_valence = st.number_input('Audio Valence', min_value=0.00, max_value=1.00)
    button = st.button('Predict')

    #If button is clilcked
    if button:
        result = predict(song_duration_ms, acousticness, danceability, energy, instrumentalness, key,
                         liveness, loudness, audio_mode, speechiness, tempo, time_signature, audio_valence)
        if result == 'Popular':
            st.success(f'This song is {result}')
        else:
            st.warning(f'This song is {result}')

def predict(song_duration_ms, acousticness, danceability, energy, instrumentalness, key,
            liveness, loudness, audio_mode, speechiness, tempo, time_signature, audio_valence):
    
    # Making prediction (process user input)
    dur = 0 if song_duration_ms < 200000 else 1
    ac = 0 if acousticness < 0.5 else 1
    dance = 0 if danceability < 0.5 else 1
    en = 0 if energy < 0.5 else 1
    ins = 0 if instrumentalness < 0.5 else 1
    key = 0 if key == [3,4,6,8,10] else 1
    live = 0 if liveness < 0.5 else 1
    loud = 0 if loudness < -10 else 1
    aum = 0 if audio_mode == 0 else 1
    spe = 0 if speechiness < 0.5 else 1
    tem = 0 if tempo < 120 else 1
    ts = 1 if time_signature == 4 else 0
    auv = 0 if audio_valence < 0.5 else 1

    prediction = Elastic_Net_Model.predict([[
        dur, ac, dance, en, ins, key, live, loud, aum, spe, tem, ts, auv
    ]])

    if prediction[0] >= 85:
        result = 'Very Popular'
    elif prediction[0] >= 70:
        result = 'Popular'
    elif prediction[0] >= 50:
        result = 'Quite Popular'
    elif prediction[0] >= 30:
        result = 'Less Popular'
    else:
        result = 'Not Popular'
    return result

if __name__ == "__main__":
    main()
