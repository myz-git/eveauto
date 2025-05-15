import pyttsx3
import logging

def speak(text, rate=220):
    """使用pyttsx3进行语音合成"""
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    huihui_voice = next((voice for voice in voices if 'Huihui' in voice.name), None)
    if huihui_voice:
        engine.setProperty('voice', huihui_voice.id)
    engine.setProperty('rate', rate)
    engine.say(text)
    engine.runAndWait()

def safe_speak(text, rate=220):
    """封装语音调用，记录异常"""
    try:
        speak(text, rate)
    except Exception as e:
        print( f"语音错误: {e}")

if __name__ == "__main__":
    speak("你好，这是一段测试语音。")