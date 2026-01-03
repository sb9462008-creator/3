from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import speech_recognition as sr
recognizer=sr.Recognizer()
with sr.Microphone() as source:
    print('Clearing background noise...')
    recognizer.adjust_for_ambient_noise(source, duration=2)  # Increased duration for better noise adjustment
    
    # Adjust recognition settings
    recognizer.energy_threshold = 300  # Minimum audio energy to consider for recording
    recognizer.dynamic_energy_threshold = True  # Automatically adjust for ambient noise
    recognizer.pause_threshold = 0.8  # Seconds of silence before considering the phrase complete
    
    print('\n=== Ready to listen! Please speak clearly... ===')
    print('You have 5 seconds to start speaking...')
    try:
        recordedaudio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
        print('Done recording...')
    except sr.WaitTimeoutError:
        print("No speech detected within the timeout period. Please try again.")
        exit()

try:
    print('Recognizing your speech...')
    text=recognizer.recognize_google(recordedaudio,language='en-US')
    if text:
        print('Your message: "{}"'.format(text))
        
        print('Analyzing sentiment...')
        Sentence=[str(text)]
        analyser=SentimentIntensityAnalyzer()
        for i in Sentence:
            v=analyser.polarity_scores(i)
            print('Sentiment scores:')
            print('Positive:', v['pos'])
            print('Neutral:', v['neu'])
            print('Negative:', v['neg'])
            print('Overall:', v['compound'])
except sr.UnknownValueError:
    print("Sorry, I couldn't understand what you said. Please try again.")
except sr.RequestError as e:
    print("Sorry, there was an error with the speech recognition service. Error: {}".format(e))
except Exception as ex:
    print("An unexpected error occurred:", ex)