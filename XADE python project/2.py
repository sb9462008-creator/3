import os
import sounddevice as sd
from scipy.io.wavfile import write
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# 📁 Фолдерууд үүсгэх
os.makedirs("data/audio", exist_ok=True)
os.makedirs("models", exist_ok=True)

# 🎙 WAV файлууд үүсгэх (сургалтанд зориулсан)
def generate_training_wav_files(emotions=["happy", "sad", "angry"], samples_per_emotion=3, duration=2, sr=22050):
    for emotion in emotions:
        for i in range(samples_per_emotion):
            filename = f"data/audio/{emotion}_{i+1}.wav"
            print(f"🎙 [{emotion.upper()}] {i+1}-р бичлэг эхэллээ...")
            audio = sd.rec(int(duration * sr), samplerate=sr, channels=1)
            sd.wait()
            write(filename, sr, audio)
            print("✅ Хадгалагдлаа:", filename)

# 🎙 Дуу бичих (массив хэлбэрээр)
def record_audio_array(duration=3, sr=22050):
    print("🎙 Дуу бичиж байна...")
    recording = sd.rec(int(duration * sr), samplerate=sr, channels=1)
    sd.wait()
    print("✅ Дуу бичигдлээ!")
    return recording

# 🧩 MFCC гаргах (.wav файл)
def extract_mfcc_from_file(file_path, n_mfcc=40):
    try:
        audio, sr = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"❌ MFCC гаргах үед алдаа гарлаа ({file_path}):", e)
        return None

# 🧩 MFCC гаргах (массив)
def extract_mfcc_from_array(audio, sr=22050, n_mfcc=13):
    y = np.squeeze(audio)
    y, _ = librosa.effects.trim(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc.T, axis=0).reshape(1, -1)

# 🧠 Загвар сургах
def train_model(data_dir="data/audio", model_path="models/model.pkl"):
    features = []
    labels = []

    for file in os.listdir(data_dir):
        if file.endswith(".wav") and "_" in file and not file.startswith("recorded"):
            label = file.split("_")[0].strip().lower()
            path = os.path.join(data_dir, file)
            mfcc = extract_mfcc_from_file(path)
            if mfcc is not None and not np.isnan(mfcc).any():
                features.append(mfcc)
                labels.append(label)
            else:
                print("⚠️ MFCC алдаатай эсвэл хоосон:", file)

    if len(features) < 5:
        print("⚠️ Сургалт хийхэд хангалттай WAV файл алга.")
        return None

    X = pd.DataFrame(features)
    y = pd.Series(labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print("✅ Загвар амжилттай сурлаа:", model_path)
    return model

# 🔍 Таамаг гаргах (массив)
def predict_from_array(model, features):
    probabilities = model.predict_proba(features)[0]
    emotions = model.classes_
    best_emotion = emotions[np.argmax(probabilities)]
    return emotions, probabilities, best_emotion

# 📊 Визуал үр дүн
def plot_emotions(emotions, probabilities):
    plt.figure(figsize=(8, 4))
    plt.bar(emotions, probabilities, color='skyblue')
    plt.title("🎧 Сэтгэл хөдлөлийн магадлал")
    plt.ylabel("Магадлал")
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

# 🚀 Үндсэн гүйцэтгэгч
if __name__ == '__main__':
    # 0. WAV файлууд үүсгэх (анх удаа бол)
    generate_training_wav_files(
        emotions=["happy", "sad", "angry", "neutral"],
        samples_per_emotion=3,
        duration=3
    )

    # 1. Загвар сургах
    model_path = "models/model.pkl"
    model = train_model(model_path=model_path)

    if model:
        # 2. Дуу бичих
        audio_array = record_audio_array()

        # 3. MFCC гаргах
        features = extract_mfcc_from_array(audio_array)

        # 4. Таамаг гаргах
        emotions, probabilities, best_emotion = predict_from_array(model, features)

        # 5. Үр дүн хэвлэх
        print("\n🎧 Танай дуу хоолойн сэтгэл хөдлөлийн хуваарь:")
        for emo, prob in zip(emotions, probabilities):
            print(f"  {emo.upper():<8}: {prob * 100:.2f}%")

        print(f"\n🧠 Нийт дүгнэлт: {best_emotion.upper()} сэтгэл хөдлөл давамгай байна.")
        plot_emotions(emotions, probabilities)