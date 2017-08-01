import re
from tflearn.data_utils import pad_sequences

r2 = r'https://[a-zA-Z0-9.?/&=:]*'
r = r = '[\s！？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+'
line = re.sub(r2, '', "$$$  測  試https://zhidao.baidu.com/question/1830964875242219220.html?123=123")
line = re.sub(r, '', line)
print(line)

