from cnocr import CnOcr

img_fp = 'pic/2024-06-09_022628.png'
ocr = CnOcr()  # 所有参数都使用默认值
out = ocr.ocr(img_fp)

print(out)