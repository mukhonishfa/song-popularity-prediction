import streamlit as st
import streamlit.components.v1 as stc
import pickle
import pandas as pd
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

# One-Hot Encoding
def one_hot_encode_key(selected_key):
    encoded = {}
    for i in range(12):
        encoded[f'key_{i}'] = [1 if selected_key == i else 0]
    return encoded

def one_hot_encode_time_signature(selected_ts):
    encoded = {}
    for i in range(6):
        encoded[f'time_signature_{i}'] = [1 if selected_ts == i else 0]
    return encoded

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
    liveness = right.number_input('Liveness', min_value=0.00, max_value=1.00, step=0.01)
    loudness = left.number_input('Loudness', min_value=-60.00, max_value=0.00, step=0.01)
    speechiness = right.number_input('Speechiness', min_value=0.00, max_value=1.00, step=0.01)
    tempo = left.number_input('Tempo', min_value=0.00, max_value=1.00, step=0.01)
    audio_valence = right.number_input('Audio Valence', min_value=0.00, max_value=1.00, step=0.01)
    audio_mode = left.selectbox('Audio Mode', [0,1])
    selected_key = right.selectbox('Key', list(range(12)))
    selected_ts = st.selectbox('Time Signature', list(range(6)))
    button = st.button('Predict')

    # Making Dictionary
    features = {
        'song_duration_ms':[song_duration_ms],
        'acousticness':[acousticness],
        'danceability':[danceability],
        'energy':[energy],
        'instrumentalness':[instrumentalness],
        'liveness':[liveness],
        'loudness':[loudness],
        'speechiness':[speechiness],
        'tempo':[tempo],
        'audio_valence':[audio_valence],
        'audio_mode':[audio_mode]
    }

    key_encoded = one_hot_encode_key(selected_key)
    time_signature_encoded = one_hot_encode_time_signature(selected_ts)

    # Combine all features into one dictionary
    data_dict = {**features, **key_encoded, **time_signature_encoded}

    # Meaking DataFrame from dictionary
    data = pd.DataFrame(data_dict)

    # If button is clilcked
    if st.button:
        if data.shape[1] == 26:
            # Load scaler and model
            with open('scaler.pkl', 'rb') as file:
                scaler = pickle.load(file)
            with open('best_lasso_model.pkl', 'rb') as file:
                best_lasso_model = pickle.load(file)
        
        # Transformation with scaler
        data_scaled = std.transform(data)

        # Making prediction
        prediction = best_lasso_model.predict(data_scaled)
        st.success(f'This song is {prediction[0]:.2f}')

if __name__ == "__main__":
    main()
