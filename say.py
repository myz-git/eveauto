import pyttsx3

def speak(text,rate=220):
    # 初始化语音引擎
    engine = pyttsx3.init()

    # 获取并设置语音
    voices = engine.getProperty('voices')
    huihui_voice = next((voice for voice in voices if 'Huihui' in voice.name), None)
    if huihui_voice:
        engine.setProperty('voice', huihui_voice.id)
    else:
        print("Microsoft Huihui Desktop voice not found, using default.")

    # 设置语速
    engine.setProperty('rate', rate)

    # 文本转语音
    engine.say(text)
    engine.runAndWait()

# 使用函数
if __name__ == "__main__":
    speak("你好，这是一段测试语音。")
