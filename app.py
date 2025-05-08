import streamlit as st
import streamlit.components.v1 as stc
import pickle
import numpy as np

with open('best_lasso_model.pkl', 'rb') as file:
    best_lasso_model = pickle.load(file)

with open ('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

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
    song_duration_ms = left.number_input('Song Duration (ms)', min_value=0, max_value=600000, step=1000)
    acousticness = right.number_input('Acousticness', min_value=0.00, max_value=1.00, step=0.01)
    danceability = left.number_input('Danceability', min_value=0.00, max_value=1.00, step=0.01)
    energy = right.number_input('Energy', min_value=0.00, max_value=1.00, step=0.01)
    instrumentalness = left.number_input('Instrumentalness', min_value=0.00, max_value=1.00, step=0.01)
    key = right.number_input('Key (0-11)', min_value=0, max_value=11, step=1)
    liveness = left.number_input('Liveness', min_value=0.00, max_value=1.00, step=0.01)
    loudness = right.number_input('Loudness', min_value=-60.00, max_value=0.00, step=0.01)
    audio_mode = left.selectbox('Audio Mode', [0,1])
    speechiness = right.number_input('Speechiness', min_value=0.00, max_value=1.00, step=0.01)
    tempo = left.number_input('Tempo', min_value=0.00, max_value=1.00, step=0.01)
    time_signature = right.selectbox('Time Signature', [1,2,3,4,5])
    audio_valence = st.number_input('Audio Valence', min_value=0.00, max_value=1.00, step=0.01)
    button = st.button('Predict')

    # If button is clilcked
    if button:
        data = pd.DataFrame([[song_duration_ms, acousticness, danceability, energy, instrumentalness, key,
                              liveness, loudness, audio_mode, speechiness, tempo, time_signature, audio_valence]],
                            columns=['song_duration_ms', 'acousticness', 'danceability', 'energy', 'instrumentalness', 'key',
                             'liveness', 'loudness', 'audio_mode', 'speechiness', 'tempo', 'time_signature', 'audio_valence']) 
        
        # Transformation with scaler
        data_scaled = scaler.transform(data)

        # Making prediction
        prediction = best_lasso_model.predict(data_scaled)
        st.success(f'This song is {prediction[0]:.2f}')


if __name__ == "__main__":
    main()
