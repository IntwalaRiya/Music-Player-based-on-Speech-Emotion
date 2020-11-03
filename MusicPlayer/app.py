from flask import Flask, render_template, request
import random
from IPython.display import Audio
import os
from tensorflow import keras
import random
import pickle
import soundfile
import librosa
import numpy as np
import pyaudio
import wave

app = Flask(__name__)
static = os.path.join(app.root_path, 'static')

def extract_features(path):
    # ZCR
    data, sample_rate = librosa.load(path, duration=5, offset=0.6)

    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result=np.hstack((result, zcr)) # stacking horizontally

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft)) # stacking horizontally

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc)) # stacking horizontally

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms)) # stacking horizontally

    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel)) # stacking horizontally
    
    return result
def songlist(start,end):
    song = []
    for i in range(0,6):
        songname = ''
        songname += str(random. randint(start,end)) + '.wav'
        song.append(songname)
    return song

@app.route('/')  
def upload():
    
    songnamelist = []

    for i in range(0,2):
        songname = ''
        songname += str(random. randint(1,100)) + '.wav'
        songnamelist.append(songname)

        songname = ''
        songname += str(random. randint(201,300)) + '.wav'
        songnamelist.append(songname)

        songname = ''
        songname += str(random. randint(401,500)) + '.wav'
        songnamelist.append(songname)
        
    sad = songlist(201,300)
    party = songlist(401,500)
    happy = songlist(1,100)

    return render_template("file_upload_form.html", songnamelist=songnamelist, len = len(songnamelist), leng = len(happy), happy=happy, sad=sad, party=party)

@app.route('/success', methods = ['POST'])  
def success():  
    if request.method == 'POST':  
        f = request.files['file']  
        # extract features and reshape it (less to install yet)
        features = extract_features(f).reshape(1,-1)

        model = pickle.load(open("finalized_model.sav", "rb"))

        result = model.predict(features)[0]
        mysonglist = []
        
        if(result == 'happy'):
            mysonglist = songlist(1,100)
        
        elif(result == 'surprise'):
            mysonglist = songlist(401,500)

        else:
            mysonglist = songlist(201,300)

        sad = songlist(201,300)
        party = songlist(401,500)
        happy = songlist(1,100)

        return render_template("success.html", mysonglist=mysonglist, len = len(mysonglist), leng = len(happy), happy=happy, sad=sad, party=party)

if __name__ == '__main__':  
    app.run(debug = True)  