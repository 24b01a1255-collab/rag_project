import speech_recognition as sr


def listen_voice():

    recognizer = sr.Recognizer()

    with sr.Microphone() as source:

        print("Listening...")

        recognizer.adjust_for_ambient_noise(
            source
        )

        audio = recognizer.listen(
            source
        )

    try:

        text = recognizer.recognize_google(
            audio,
            language="en-US"
        )

        return text

    except Exception:

        return None