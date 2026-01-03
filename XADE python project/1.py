import sounddevice as sd
import librosa
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.ensemble import RandomForestClassifier

#  Дуу бичих функц
def record_audio(duration=4, sr=22050):
    print(" Дуу бичиж байна...")
    recording = sd.rec(int(duration * sr), samplerate=sr, channels=1)
    sd.wait()
    print(" Дуу бичигдлээ!")
    return recording

#  MFCC шинж чанар гаргах
def extract_mfcc(audio, sr=22050, n_mfcc=13):
    y = np.squeeze(audio)
    y, _ = librosa.effects.trim(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc.T, axis=0).reshape(1, -1)
    return mfcc_mean

# 🌲 Загвар сургах ба хадгалах
def train_and_save_model(path="emotion_model.pkl"):
    X_train = np.random.rand(12, 13)
    y_train = np.array(["happy", "sad", "angry", "calm", "happy", "angry",
                        "sad", "happy", "calm", "angry", "sad", "happy"])
    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, path)
    print(f" Загвар хадгалагдлаа: {path}")

def load_model(path="emotion_model.pkl"):
    if not os.path.exists(path):
        train_and_save_model(path)
    return joblib.load(path)

#  Таамаг гаргах
def predict_emotion(model, features):
    probabilities = model.predict_proba(features)[0]
    emotions = model.classes_
    best_emotion = emotions[np.argmax(probabilities)]
    return emotions, probabilities, best_emotion

#  Визуал үр дүн
def plot_emotions(emotions, probabilities):
    plt.figure(figsize=(8, 4))
    plt.bar(emotions, probabilities, color='skyblue')
    plt.title(" Сэтгэл хөдлөлийн магадлал")
    plt.ylabel("Магадлал")
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

#  Үндсэн гүйцэтгэгч
def main():
    model = load_model()
    audio = record_audio()
    features = extract_mfcc(audio)
    emotions, probabilities, best_emotion = predict_emotion(model, features)

    print("\n Танай дуу хоолойн сэтгэл хөдлөлийн хуваарь:")
    for emo, prob in zip(emotions, probabilities):
        print(f"  {emo.upper():<8}: {prob * 100:.2f}%")

    print(f"\n Нийт дүгнэлт: {best_emotion.upper()} сэтгэл хөдлөл давамгай байна.")
    plot_emotions(emotions, probabilities)

if __name__ == "__main__":
    main()