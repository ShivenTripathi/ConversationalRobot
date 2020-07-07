import speech_recognition as sr  

def speechInput(): 
    r = sr.Recognizer()  
    with sr.Microphone() as source:  
        print("Please wait. Calibrating microphone...")  
        r.adjust_for_ambient_noise(source, duration=1)  
        print("Say something!")  
        audio = r.listen(source)   
    try: 
        speech= r.recognize_sphinx(audio)
        print("You said: '" + r.recognize_sphinx(audio) + "'")
        return speech  
    except sr.UnknownValueError:  
        print("I could not understand audio :(")  
    except sr.RequestError as e:  
        print("Recog error; {0}".format(e))  

while input("Continue?(y/n)") is 'y':
    speech=speechInput()
    print(speech)