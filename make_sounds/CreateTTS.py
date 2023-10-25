from gtts import gTTS

text = "down"
lang = "ko"

tts = gTTS(text=text, lang=lang)

tts.save("/Users/min_leon/Desktop/졸작준비/resources/sounds/down.mp3")
