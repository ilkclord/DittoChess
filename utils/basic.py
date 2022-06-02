from lib2to3.pgen2.token import tok_name
from sre_constants import MAGIC
import sys
import numpy as np

# 0.2 0.8 for Sigmoid
MAGIC = [0.2, 0.8]
softmax = 0
# list of board
def mapping(target, value):
    map = np.zeros(64)
    while 1:
        try:
            index = target.index(value)
            map[index] = 1
            target[index] = 0
        except:
            break
    return np.reshape(map, (8, 8)).tolist()

# feature now -> 14
# piece -> 6 * 2
# allies ->  1
# enemies -> 1
def bitmap(target):
    result = []
    flat = np.reshape(target, 64).tolist()
    flat2 = np.zeros(64)
    for i in range(1, 7):
        map = mapping(flat, i)  
        map2 = mapping(flat, -1 * i)
        result.append(map)
        result.append(map2)
    # add allies , enemies
    for idx , i in enumerate(flat) :
        if i > 0 :
            flat[idx] = 1
        elif i < 0 :
            flat2[idx] = 1

    result.append(np.reshape(flat, (8, 8)).tolist())
    result.append(np.reshape(flat2, (8, 8)).tolist())
    return result
# read the data file , return board and from to (list)
# seq : desired time sequence
# bit + seq = 1 -> single board , for Conv2D
# time-bit + seq != 1 -> mutiple board , for Conv3D


def read_file(file_name , trans = "" , seq = 1) :
    file = open(file_name , 'r')
    line = file.readline()
    checkmate= line
    line = file.readline()
    oposi , oposj , nposi , nposj  = line.split()
    from_ans = np.full((8,8) , MAGIC[0])
    to_ans = np.full((8, 8), MAGIC[0])
    from_ans[int(oposi)][int(oposj)] = MAGIC[1]
    to_ans[int(nposi)][int(nposj)] = MAGIC[1]
    arr = [] # 3D (algebra) or 4D (bitmap)
    current = []
    while seq > 0 :
        _arr = []
        for i in range(0,8) :
            line = file.readline()
            tmp = line.split()
            if tmp == [] :
                break
            __arr = []
            for a in tmp :
                __arr.append(int(a))
            _arr.append(__arr)
        if _arr == [] :
            break
        if arr == [] :
            current = _arr
        if trans != "" :
            _arr = bitmap(_arr)
        arr.append(_arr)
        seq = seq - 1
    file.close()
    if seq != 0 :
        if trans != "" :
            for i in range(seq) :
                arr.append(np.zeros((14,8,8)).tolist())
        else :
            for i in range(seq) :
                arr.append(np.zeros((8,8)).tolist())
    # Apply when using softmax
    if softmax :
        for i , c in enumerate(current) :
            for j, _c in enumerate(c):
                if _c <= 0:
                    from_ans[i][j] = MAGIC[0]*0.05

    if trans == "bit" :
        return arr[0] ,from_ans , to_ans , current
    elif trans == "time-bit" :
        return arr , from_ans, to_ans, current
    else :
        return arr, from_ans, to_ans , []
def read_file_lstm(file_name , seq) :
    file = open(file_name, 'r')
    line = file.readline()
    oposi, oposj, nposi, nposj = line.split()
    from_ans = np.full((8, 8), MAGIC[0])
    to_ans = np.full((8, 8), MAGIC[0])
    from_ans[int(oposi)][int(oposj)] = MAGIC[1]
    to_ans[int(nposi)][int(nposj)] = MAGIC[1]
    arr = []  # 3D
    for i in range(0,seq):
        _arr = []
        for i in range(0, 8):
            line = file.readline()
            if line == "":
                break
            tmp = line.split()
            __arr = []
            for a in tmp:
                __arr.append(int(a))
            _arr.append(__arr)
        if _arr == []:
            break
        arr.append(_arr)
    if len(arr) < seq :
        arr.insert(0 , arr[0])
    file.close()
    return arr, from_ans, to_ans
# col
# Get max-sorted index list

#Top n
def getnmax(target , number) :
    if len(target) < number :
        return None
    result = []
    tmp  = target.copy() 
    tmp2 = target.copy()
    tmp.sort(reverse = True)
    for idx ,i in enumerate(tmp) :
        if idx == number :
            break
        id = tmp2.index(i)
        result.append(id)
        tmp2[id] = -1
    return result 

#Max-sort index
def getmaxn(target):
    result = []
    tmp = target.copy()
    tmp2 = target.copy()
    tmp.sort(reverse=True)
    for idx, i in enumerate(tmp):
        id = tmp2.index(i)
        result.append(id)
        tmp2[id] = -1
    return result
