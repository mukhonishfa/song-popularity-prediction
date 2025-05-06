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
                Kaggle: Link <https://www.kaggle.com/datasets/yasserh/song-popularity-dataset>
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
    acousticness = right.number_input('Acousticness', min_value=0.000000, max_value=1.000000)
    danceability = left.number_input('Danceability', min_value=0.000, max_value=1.000)
    energy = right.number_input('Energy', min_value=0.000, max_value=1.000)
    instrumentalness = left.number_input('Instrumentalness', min_value=0.000000, max_value=1.0)
    key = right.number_input('Key', min_value=0, max_value=11)
    liveness = left.number_input('Liveness', min_value=0.0000, max_value=1.0000)
    loudness = right.number_input('Loudness', min_value=-15.000, max_value=1.000)
    audio_mode = left.selectbox('Audio Mode',(1,0))
    speechiness = right.number_input('Speechiness', min_value=0.0000, max_value=1.000)
    tempo = left.number_input('Tempo', min_value=50.000, max_value=200.000)
    time_signature = right.number_input('Time Signature', min_value=0, max_value=5)
    audio_valence = st.number_input('Audio Valence', min_value=0.0000, max_value=1.000)
    button = st.button('Predict')

    #If button is clilcked
    if button:
        result = predict(song_duration_ms, acousticness, danceability, energy, instrumentalness, key,
                         liveness, loudness, audio_mode, speechiness, tempo, time_signature,audio_valence)
        if result == 'Popular':
            st.success(f'This song is {result}')
        else:
            st.warning(f'This song is {result}')

def predict(song_duration_ms, acousticness, danceability, energy, instrumentalness, key,
            liveness, loudness, audio_mode, speechiness, tempo, time_signature,audio_valence)
    
    # Making prediction (process user input)
    prediction = Elastic_Net_Model.predict([[
        song_duration_ms, acousticness, danceability, energy, instrumentalness, key,
        liveness, loudness, audio_mode, speechiness, tempo, time_signature,audio_valence
    ]])

    result = 'Not Popular' if prediction == 0 else 'Popular'
    return result

if __name__ == "__main__":
    main()