from gtts import gTTS

text = "자세가 무너졌습니다"
lang = "ko"

tts = gTTS(text=text, lang=lang)

tts.save("../resources/sounds/broken_posture.mp3")
