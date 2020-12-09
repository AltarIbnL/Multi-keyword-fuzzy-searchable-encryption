import hashlib
import numpy as np
import cryptography
import  Cryptodome
from pybloom_live import ScalableBloomFilter, BloomFilter
'''
对偶编码函数，将字符转成向量形势

给定字符串 Ｓ_１ ＝ C_1C_2C_3....C_n
和二进制向量 Ｓ ２ ＝ ｂ０ｂ１ … ｂ_(ｍ－１) ， 
Ｓ ２ 中每位元素的初始值为０ ，其中 ｎ ＜ ｍ ．通过 Ｈａｓｈ函数 Ｈ ，
把 Ｓ１中相邻２个字符散列映射为０～ （ ｍ －１ ）之间的数，
当且仅当 Ｈ(Ｃ_ｊ ， Ｃ_{ｊ＋１} ) ＝ ｉ 时， ｂ_ｉ ＝１

'''
def Dual_code(S1):

    temp = []
    for index,value in enumerate(S1):
        x = 0
        temp1 = [x for i in range(676)]  #建立一个全为0的数组
        # print(value)
        length=len(value)
        for i,v in enumerate(value):  # 遍历当前关键字

            if(i==length-1):
                C = ord(v)+ord(' ') #如果是最后一个字符，将最后一个字符转换成ascii相加
                C = C%676
                while(temp1[C]==1):  #将hash表中对应位置置1，如果该位置已经有1，则后移
                    if(C!=675):
                        C+=1
                    else:
                        C=0
                temp1[C] = 1

            else:
                C = ord(v)+ord(S1[index][i+1])
                C = C%676

                while (temp1[C] == 1):
                    if (C != 675):
                        C += 1
                    else:
                        C = 0
                temp1[C] = 1
        temp.append(temp1)
    return temp



if __name__ == '__main__':
    S1=['jugment ae12','kugou']
    S2=Dual_code(S1)
    print(S2[0].count(1))
    print(S2[0])
    k=np.random.randint(0,1,128)

    bloom = BloomFilter(capacity=1024, error_rate=0.01)
    bloom.add(S2[0])

    print(S2[0] in bloom)