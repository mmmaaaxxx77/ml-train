import jieba
import numpy as np
import re
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import HashingVectorizer
from tflearn.data_utils import pad_sequences

r2 = r'https://[a-zA-Z0-9.?/&=:]*'
r = r = '[\s！？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+'
line = re.sub(r2, '', "$$$  測  試https://zhidao.baidu.com/question/1830964875242219220.html?123=123")
line = re.sub(r, '', line)
print(line)

comma_tokenizer = lambda x: jieba.cut(x, cut_all=False, HMM=True)

vect = FeatureHasher(n_features=2**21, non_negative=True)

terms = comma_tokenizer("柯P是柯市長")

terms = [{t: 1} for t in terms]
print(terms)
print(vect.transform(terms)[0].nonzero()[1][0])
#print(vect.transform(terms).toarray())


def transform_term_to_id(sentence):
    terms = comma_tokenizer(sentence)
    terms = [{t: 1} for t in terms]
    ids_sci = vect.transform(terms)
    return np.asarray([item.nonzero()[1][0] for item in ids_sci])

print(transform_term_to_id("柯P柯P是柯市長"))
print(transform_term_to_id("柯P柯P是柯市長").size)
