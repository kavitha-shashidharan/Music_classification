import pickle
import random
import string
import warnings

import numpy as np
import pandas as pd
import flask
import librosa
import matplotlib.pyplot as plt
import youtube_dl
from flask import Flask, request, render_template
import dill
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings('ignore')
app = Flask(__name__)

# Opening the pickle file

pipe = pickle.load(open('pipe.pkl', 'rb'))


@app.route('/')
def index():
    return flask.render_template('music.html')

@app.route('/result', methods=['POST'])
def predict():
    if flask.request.method == 'POST':
        input = flask.request.form
        url = input.get('youtube')


# Downloading the mp3 file using youtube url

        letters = string.ascii_lowercase
        # Generating random string to assign a filename
        songfile = ''.join(random.choice(letters) for i in range(10))
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': '{}.%(ext)s'.format(songfile),
            'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',}]}
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        audio_path = songfile + '.mp3'

# Using librosa to get the audio file and extract the features

        y, sr = librosa.load(audio_path, duration = 30)
        cmap = plt.get_cmap('inferno')
        plt.specgram(y, NFFT=2048, Fs=2, Fc=0, noverlap=128, cmap=cmap, sides='default', mode='default', scale='dB');
        plt.axis('off');
        plt.savefig(f'{audio_path}.png')
        plt.clf()
        chroma_stft = (librosa.feature.chroma_stft(y=y, sr=sr))
        rmse = (librosa.feature.rms(y = y))
        spec_cent = (librosa.feature.spectral_centroid(y=y, sr=sr))
        spec_bw = (librosa.feature.spectral_bandwidth(y=y, sr=sr))
        rolloff = (librosa.feature.spectral_rolloff(y=y, sr=sr))
        zcr = (librosa.feature.zero_crossing_rate(y))
        mfcc = (librosa.feature.mfcc(y=y, sr=sr))
        a = []
        for e in mfcc:
            a.append(f'{np.mean(e)}')
        #result = flask.request.form
        new = pd.DataFrame({
            'chroma_stft' : [np.mean(chroma_stft)], 'rmse' : [np.mean(rmse)],
            'spectral_centroid' : [np.mean(spec_cent)], 'spectral_bandwidth' : [np.mean(spec_bw)],
            'rolloff' : [np.mean(rolloff)], 'zero_crossing_rate' : [np.mean(zcr)], 'mfcc1' : [a[0]],
            'mfcc2' : [a[1]], 'mfcc3' : [a[2]], 'mfcc4' : [a[3]], 'mfcc5' : [a[4]],
            'mfcc6' : [a[5]], 'mfcc7' : [a[6]], 'mfcc8' : [a[7]], 'mfcc9' : [a[8]],
            'mfcc10' : [a[9]], 'mfcc11' : [a[10]], 'mfcc12' : [a[11]], 'mfcc13' : [a[12]],
            'mfcc14' : [a[13]], 'mfcc15' : [a[14]], 'mfcc16' : [a[15]], 'mfcc17' : [a[16]],
            'mfcc18' : [a[17]], 'mfcc19' : [a[18]], 'mfcc20' : [a[19]]}, index = [0])

        yhat = pipe.predict(new)[0]
        #yhat = yhat.round()
        if yhat == 0:
            pred = 'HAPPY!! - This is gonna cheer you up'
        elif yhat == 1:
            pred = 'SAD - Guess melancholy has a deeper essence here'
        return flask.render_template('music.html', prediction=pred)

if __name__ == '__main__':
    app.run(debug=True)
