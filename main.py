
import random
import numpy as np
import os
from collections import Counter #used to count most common word
from collections import defaultdict
from dualcode import Dual_code
import e2LSH as LSH
import bloomfilter as BF

def keygen(k):
    '''
    # 输入参数k，返回密钥sk
    :param k: 随机矩阵的维度
    :return: sk=(M1,M2,S)
    '''
    M1=np.random.randint(0,2,[k,k])
    M2=np.random.randint(0,2,[k,k])
    S=np.random.randint(0,2,[k])
    sk = [M1,M2,S]

    return sk

def BuildIndex(sk,F,W):
    '''

    param: sk=(M1,M2,S), F是文档集合，W={w1,w2....,wn}是文档对应的关键字
    return：I=(F,I',I'')

    '''
    V=[]  #V储存关键字转换成编码后的值
    # 为每个文件生成布隆过滤器
    B=[BF.BloomFilter(capacity=1024,error_rate=0.001) for i in range(len(W))]

    for val in W:
        # Dual_code是对偶编码函数，输入一个数组，将数组内的关键字转换成向量
        v_i = Dual_code(val)
        V.append(v_i)

    # 将向量集合V中每一个向量vi，通过位置敏感函数Hash进行计算,注意，此时vi每一个元素都是字符串，后面需要将其转换成浮点型
    for idx,val in enumerate(V):  # 遍历每一个文档对应关键字的数据向量
        '''
        e2LSH是符合P-STABLE分布的哈希敏感函数，
        p返回4个元素，分别是①哈希列表，里面储存有20个（自行设定）哈希函数
        ②hashFun哈希功能表，储存5个（自行设定）哈希列表，③fpRand 随机数集合，④hash_collect 每个数据向量hash多次后后的值
        '''
        p=LSH.e2LSH(val,k=20, L=5, r=1, tableSize=20)

        H_p=p[3] #提取每个数据向量hash 20次 后的值
        # 将结果插入布隆过滤器
        for j in range(len(H_p)):
            # print(H_p[j])
            B[idx].add(H_p[j])  # 逐个数据向量输入bloom
    S = sk[2]  #提取私钥中的S
    r = random.randint(1,10) # 生成一个随机数
    # B_i = {b1,b2...bk} 每个文档对应哈希列表的每一位
    # 遍历S和B中的每一位，如果S中对应的s_j=1，则b_j'=b_j''=b_j；若s_j=0，令b_j' = 1/2*b_j+r 和b_j''=1/2*b_j-r。
    B_P=[]
    B_DP=[]  #B'、B''
    I=[F]  # 索引
    for i,b in enumerate(B):
        for j,vals in enumerate(S):
            if vals==1:
                # 如果s_j=1
                # print(j)
                b_jp=b_jpp=int(b.bitarray[j])  # b_j'=b_j''=b_j ,v是第i个bloomfilter
                B_P.append(b_jp)
                B_DP.append(b_jpp)
            else:
                # 如果s_j=0
                b_jp = 0.5*int(b.bitarray[j])+r
                b_jpp = 0.5*int(b.bitarray[j])-r
                B_P.append(b_jp)
                B_DP.append(b_jpp)

        I_P = np.inner(np.transpose([sk[0]]).reshape((128,128)),B_P) # 将M1矩阵转置后与B'点乘
        I_DP = np.inner(np.transpose([sk[1]]).reshape((128,128)),B_DP) # 将M2矩阵转置后与B''点乘
        I_i = [I_P,I_DP]
        I.append(I_i)
        B_DP = []  # 清空B'和B''
        B_P = []

    return I    # I是由多个文件夹构成的索引组成

def initial_key(F):  # 生成每个文档对应的关键字结合
    '''
        param F :文档集合
        return: W={w1,w2,w3.....wn}  文档对应关键字集合

    '''
    W=[]
    filedata=[]
    for idx,val in enumerate(F):
        # print(idx)
        cnt = Counter()
        for line in open('./doc/'+F[idx],'r'):
            word_list = line.replace(',', '').replace('\'', '').replace('.', '').lower().split()
            # print(word_list)
            for word in word_list:
                cnt[word]+=1  #统计一个文档中各个关键字出现次数
        filedata.append((val, cnt))  #获得文档列表所有单词的频率


    for idx, val in enumerate(filedata):
        allwords = []  # 储存不同文档的频率最高的单词
        for value, count in val[1].most_common(10):  #找到每个文档频率最高的5个单词
            if value not in allwords:
                allwords.append(value)
            # print(allwords)
        W.append(allwords)   # W 储存的是文档对应的关键字
    # print(W)
    return W
    # print(filedata[0])

def Trapdoor(sk,Q):
    '''
    :param sk: 密钥
    :param Q:  查询关键字
    :return: t={t',t''}
    '''
    V = []  # V储存关键字转换成编码后的值
    # 生成1个布隆过滤器
    B_t = BF.BloomFilter(capacity=1024, error_rate=0.001)

    # Dual_code是对偶编码函数，输入一个数组，将数组内的关键字转换成向量。此时查询关键字集合相当于只在一个文档内部，无需循环
    v_i = Dual_code(Q)
    V.append(v_i)

    # 将向量集合V中每一个向量vi，通过位置敏感函数Hash进行计算,注意，此时vi每一个元素都是字符串，后面需要将其转换成浮点型
    for idx,val in enumerate(V):  # 遍历每一个文档对应关键字的数据向量
        '''
        e2LSH是符合P-STABLE分布的哈希敏感函数，
        p返回4个元素，分别是①哈希列表，里面储存有20个（自行设定）哈希函数
        ②hashFun哈希功能表，储存5个（自行设定）哈希列表，③fpRand 随机数集合，④hash_collect 每个数据向量hash多次后后的值
        '''
        p=LSH.e2LSH(val,k=20, L=5, r=1, tableSize=20)

        H_p=p[3] #提取每个数据向量hash 20次 后的值
        # 将结果插入布隆过滤器
        for j in range(len(H_p)):
            # print(H_p[j])
            B_t.add(H_p[j])  # 逐个数据向量输入bloom
    S = sk[2]  # 提取私钥中的S
    r = random.randint(1, 10)  # 生成一个随机数
    # B_i = {b1,b2...bk} 每个文档对应哈希列表的每一位
    # 遍历S和B中的每一位，如果S中对应的s_j=0，则b_j'=b_j''=b_j；若s_j=1，令b_j' = 1/2*b_j+r 和b_j''=1/2*b_j-r。
    B_P = []
    B_DP = []  # B'、B''

    for j,vals in enumerate(S):
        if vals==0:
            # 如果s_j=0
            # print(j)
            b_jp=b_jpp=int(B_t.bitarray[j])  # b_j'=b_j''=b_j ,v是第i个bloomfilter
            B_P.append(b_jp)
            B_DP.append(b_jpp)
        else:
            # 如果s_j=1
            b_jp = 0.5*int(B_t.bitarray[j])+r
            b_jpp = 0.5*int(B_t.bitarray[j])-r
            B_P.append(b_jp)
            B_DP.append(b_jpp)

    T_P = np.inner(np.linalg.inv([sk[0]]).reshape((128, 128)), B_P)  # 将M1矩阵取逆后与B'点乘
    T_DP = np.inner(np.linalg.inv([sk[1]]).reshape((128, 128)), B_DP)  # 将M2矩阵取逆后与B''点乘
    t=[T_P,T_DP]
    return t

def Search(I,t):
    '''

    :param I:  文档索引
    :param t:  搜索陷门
    :return: File 包含关键字的文档集合
    '''
    File=[]
    for I_i in I[1:]:
        I_iP = I_i[0]  # I_i'
        I_iDP =  I_i[1] # I_i''
        R_i = np.inner(I_iP,t[0])+np.inner(I_iDP,t[1])
        File.append(R_i)
    return File




if __name__ == '__main__':

    query = ['he']
    k=128
    sk=keygen(k)
    F = [x for x in os.listdir('./doc')]
    F.sort()
    W=initial_key(F)  #获得文档对应的关键字集合W\
    print(W)
    Index = BuildIndex(sk,F,W)
    trap = Trapdoor(sk,query)
    F_R = Search(Index,trap)
    F_R.sort()
    print(F_R)

    # print(W)
    # d=BF.BloomFilter(capacity=100,error_rate=0.01)
    # print(d.num_bits)
    # d.add('asd')
    # s=[int(x) for x in d.bitarray]

