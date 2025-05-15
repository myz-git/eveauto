import azure.cognitiveservices.speech as speechsdk
#from utils import load_config

def load_config():
    config = {}
    with open('static/cfg.txt', 'r') as file:
        lines = file.readlines()
        for line in lines:
            key, value = line.strip().split('=')
            config[key.strip()] = value.strip()
    return config

def setup_speech_service(style):
    # 设置语音服务的订阅密钥和区域
    config = load_config()
    speech_key = config['KEY']
    service_region = config['REGION']

    #print (speech_key,service_region)
    # 创建语音配置实例
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
    speech_config.speech_synthesis_voice_name = "zh-CN-XiaoxiaoNeural"
    

    # 创建语音合成器实例
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)
    return speech_synthesizer

def speak(text,style="assistant"):
    # 获取配置好的语音合成器
    
    speech_synthesizer = setup_speech_service(style)

    # 执行文本的语音合成
    result = speech_synthesizer.speak_text_async(text).get()

    # 检查合成结果
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        pass
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print("Speech synthesis canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("Error details: {}".format(cancellation_details.error_details))

# Example usage
if __name__ == "__main__":
    text_to_speak = "A杠B星系本地有人进入"
    speak(text_to_speak)
