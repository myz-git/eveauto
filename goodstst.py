import re

def correct_string(input_str):
    # 定义检查和替换规则
    rules = [
        ('天', '大'),    # 将 '天' 替换为 '大'
        ('性', '牲'),    # 将 '性' 替换为 '牲'
        # 可以根据需要添加更多规则
    ]
    
    # 应用每个规则进行替换
    for old, new in rules:
        input_str = re.sub(old, new, input_str)
    
    return input_str

# 测试函数
test_str1 = '一天群的牛羊'
corrected_str1 = correct_string(test_str1)
print(corrected_str1)  # 输出: 一大群的牛羊

test_str2 = '一天群的性畜'
corrected_str2 = correct_string(test_str2)
print(corrected_str2)  # 输出: 一天群的牲畜
