
from cnocr import CnOcr

img_fp = 'pic//2024-06-13_104256.png'
ocr = CnOcr() # 所有参数都使用默认值
#ocr = CnOcr(rec_model_name='densenet_lite_246-gru_base')
res = ocr.ocr(img_fp)


txt = '货物'
for line in res:
    if txt in line['text']:
        # 假设我们可以获取到文字的位置
        print(line['position'][0][0])
        print(line['position'][0][1])
        print(line['position'][1][0])
        print(line['position'][1][1])

        #x = line['position'][0][0] + (line['position'][1][0] - line['position'][0][0]) // 2
        #y = line['position'][0][1] + (line['position'][2][1] - line['position'][0][1]) // 2
            
        # 移动鼠标并点击代理人名字
        #pyautogui.moveTo(x, y)
    
        break

# 打印结果
