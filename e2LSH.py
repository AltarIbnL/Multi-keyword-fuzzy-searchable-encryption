import random
import numpy as np
# from test_helper import *

class TableNode(object):
    def __init__(self, index):
        self.val = index
        self.buckets = {}


def genPara(n, r):
    """
    # 生成构造函数簇之中所需的参数a,b，a是一个长度为n的向量。

    :param n: length of data vector
    :param r:
    :return: a, b
    """
    # 对于一个向量v（相当于上面公式中的(v1,v2,…,vn)），现在从p稳定分布中，随机选取v的维度个随机变量（即n个随机变量）
    a = []
    for i in range(n):
        a.append(random.gauss(0, 1))
    b = random.uniform(0, r)  #生成0到r之间的实数

    return a, b


def gen_e2LSH_family(n, k, r):
    """
    生成K个包含参数(a,b)的列表，
    :param n: length of data vector
    :param k:
    :param r:
    :return: a list of parameters (a, b)
    """
    result = []
    for i in range(k):
        result.append(genPara(n, r))   #获得参数ab

    return result


def gen_HashVals(e2LSH_family, v, r):
    """

    :param e2LSH_family: include k hash funcs(parameters)
    :param v: data vector
    :param r:
    :return hash values: a list
    """

    # hashVals include k values
    hashVals = []

    for hab in e2LSH_family: #hab[0]是向量a，hab[1]是b，有k=20个哈希函数

        hashVal = (np.inner(hab[0], v) + hab[1]) // r
        hashVals.append(hashVal)

    return hashVals  #包含20个哈希值


def H2(hashVals, fpRand, k, C):
    """

    :param hashVals: k hash vals
    :param fpRand: ri', the random vals that used to generate fingerprint
    :param k, C: parameter
    :return: the fingerprint of (x1, x2, ..., xk), a int value
    """
    return int(sum([(hashVals[i] * fpRand[i]) for i in range(k)]) % C)


def e2LSH(dataSet, k, L, r, tableSize):
    """
    generate hash table

    * hash table: a list, [node1, node2, ... node_{tableSize - 1}] 构建一个哈希表
    ** node: node.val = index; node.buckets = {}
    *** node.buckets: a dictionary, {fp:[v1, ..], ...}

    :param dataSet: a set of vector(list)  数据向量
    :param k: k表示hash函数的数量 与 k个数组成的整数向量
    :param L: L组hash组,L组hash函数，每组由k个hash函数构成
    :param r:
    :param tableSize:
    :return: 3 elements, hash table, hash functions, fpRand
    """
    # TableNode 是一个类，将表的编号作为其内部的值
    hashTable = [TableNode(i) for i in range(tableSize)]
    # n=len(dataSet[0]) 获得列表的列数,数据集传入5个向量,m=len(dataSet)获得行数
    n = len(dataSet[0])
    # print(n)
    m = len(dataSet)
    # print(m)
    C = pow(2, 32) - 5
    hashFuncs = []
    fpRand = [random.randint(-10, 10) for i in range(k)]   #fpRand 是用于计算一个数据向量的“指纹”随机数


    #构造函数簇，之后会从中取出k个hahs函数，而且LSH函数簇中hash函数计算公式是：h_ab=[(a▪v+b)/r]
    e2LSH_family = gen_e2LSH_family(n, k, r)

    # hashFuncs: [[h1, ...hk], [h1, ..hk], ..., [h1, ...hk]]
    # hashFuncs include L hash functions group, and each group contain k hash functions
    hashFuncs.append(e2LSH_family)
    hash_collect=[]  # 记录hash数据向量后不同数据向量的值
    for dataIndex in range(m):  # 对文档中所有向量进行运算
        # print(dataSet[dataIndex])
        # 对一个数据向量生成k个哈希值
        hashVals = gen_HashVals(e2LSH_family, dataSet[dataIndex], r)
        hash_collect.append(hashVals)
        # 生成指纹向量
        fp = H2(hashVals, fpRand, k, C)

        # generate index，也就是H1的值
        index = fp % tableSize

        # 查找到在哈希列表的节点，每个节点储存具有相同或者相近指纹向量的数据。经过H1哈希后index相同，储存在同一个桶中，就代表他们是邻近的
        node = hashTable[index]

                # node.buckets 是一个字典：{123:[1],0:[362,585]} , 123表示指纹向量，1代表第一个数据向量；同理，后面的0表示指纹向量，362,585表示数据向量的索引
        if fp in node.buckets:

            # bucket is vector list
            bucket = node.buckets[fp]

            # add the data index into bucket
            bucket.append(dataIndex)

        else:
            node.buckets[fp] = [dataIndex]

    return hashTable, hashFuncs, fpRand,hash_collect


def nn_search(dataSet, query, k, L, r, tableSize):
    """

    :param dataSet:
    :param query:
    :param k:
    :param L:
    :param r:
    :param tableSize:
    :return: the data index that similar with query
    """

    result = set()

    temp = e2LSH(dataSet, k, L, r, tableSize)
    C = pow(2, 32) - 5

    # 构建了一个hashtable和 包含L个哈希函数组，每个组内右k个哈希函数的 hashFuncGroups
    hashTable = temp[0]
    hashFuncGroups = temp[1]
    fpRand = temp[2]

    for hashFuncGroup in hashFuncGroups:
        # 每个hashFuncGroup储存的是K个哈希函数
        # get the fingerprint of query 获得查询数据向量的指纹向量
        queryFp = H2(gen_HashVals(hashFuncGroup, query, r), fpRand, k, C)

        # get the index of query in hash table，得到查询向量的H1值
        queryIndex = queryFp % tableSize

        # L个hash函数组，每个函数组包含K个hash。每次遍历一个函数组，从hashTable的字典中获取bucket，这个bucket包含与查询向量位置相近的所有数据向量的
        if queryFp in hashTable[queryIndex].buckets:
            result.update(hashTable[queryIndex].buckets[queryFp])

    return result

if __name__ == '__main__':
    C = pow(2, 32) - 5
    dataSet = readData("test_data.csv")
    query = [-2.7769, -5.6967, 5.9179, 0.37671, 1]

    indexes = nn_search(dataSet, query, k=20, L=5, r=1, tableSize=20)
    for index in indexes:   # 计算查询向量和相近数据向量的欧氏距离
        print("index:%d" % index, euclideanDistance(dataSet[index], query))