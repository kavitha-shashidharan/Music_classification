# Music_classification

This project is built to predict if a particular song is happy or sad. 

### Creating the data repository

The dataset repository was scraped from youtube by 
downloading the song and splitting it into one minute audio clips. This has been done for both happy and sad songs by using
youtube-dl and ffmpeg shell script commands. I downloaded both happy and sad piano solo instrumental pieces to create this 
dataset.

The documentation of youtube-dl can be found here - https://github.com/ytdl-org/youtube-dl

### Extracting the features using Librosa

Once the song repository has beeen created with one minute long audio clips, **Librosa** analyses the audio signals of the audio clips and gives out two important values:

```python
import librosa
audio_path = '../happy000.mp3'
x , sr = librosa.load(audio_path)
```

Here, **x** contains an audio time series in the form of numpy arrays with a default sampling rate(sr) of 22KHZ. Using these values few features of the songs like zero-crossing-rate, speactral-bandwidth are calculated. 

Check out my blog page - https://myinsightsondata.com/blog/music-classification for more details about using librosa audio library and extracting the features. The github documentation of librosa can be found here - https://librosa.github.io/librosa/

### Final steps

After extracting the featuree, I built a model using Logistic Regression to predict the mood of the song. Then built a flask app and deployed it into azure using docker. 

