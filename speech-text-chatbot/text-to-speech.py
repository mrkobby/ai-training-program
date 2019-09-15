import pyttsx3

while True:
    inp = input()

    if inp.lower() == 'quit':
        break

    engine = pyttsx3.init()
    engine.say(inp)
    engine.runAndWait()