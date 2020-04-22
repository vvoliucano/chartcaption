#coding=utf-8
import json
# import pandas as pd
import numpy as np
import numpy
import math
import random
import matplotlib.pyplot as plt

def generate_line_data(begin, end, sigma = 0.05, total_number = 100):
    data = []
    delta_x = begin[0] - end[0]
    delta_y = begin[1] - end[1]
    delta_distance = math.sqrt(delta_x * delta_x + delta_y * delta_y)
    x_unit = - delta_y / delta_distance
    y_unit = delta_x / delta_distance
    for i in range(total_number):
        rate = numpy.random.uniform(0,1)
        original_x = begin[0] + rate * (end[0] - begin[0])
        original_y = begin[1] + rate * (end[1] - begin[1]) # 我们首先搞一个初始的位置
        uniform_error = numpy.random.normal(0, sigma) # 我们接着设定一个随机变量
        data.append([original_x + uniform_error * x_unit, original_y + uniform_error * y_unit]) # 然后在直线的垂直方向加
    data = numpy.asarray(data, dtype=np.int16)
    return data

def generate_class_data(center = [0.5, 0.5], total_number = 100, sigma = [0.05, 0.05], corr = 0):
    cov_xy = corr * sigma[0] * sigma[1]
    cov = [[sigma[0] * sigma[0], cov_xy],[cov_xy, sigma[1] * sigma[1]]]
    x = np.random.multivariate_normal(center, cov, total_number)
    return x

def get_cluster_sentences(centers, amounts):
    total = len(centers)
    rank = np.asarray(amounts, dtype=np.int16).argsort()[::-1][:total]
    sentence = f'There are {total} clusters.'
    for idx in range(len(amounts)):
        c = centers[idx]
        x = int(c[0])
        y = int(c[1])
        if idx == 0:
            sentence+= f'The largest cluster is around ({x}, {y}).'
            if total > 2:
                sentence += f'Other centers are '
            else:
                sentence += f'The other one is '
        else:
            delim = '.' if idx==len(amounts)-1 else ', '
            sentence += f'({x}, {y}){delim}'

    return list(filter(None, sentence.split('.')))

def get_line_sentences(start, end):
    k = (end[1]-start[1])/(end[0]-start[0])
    if(k>0):
        sentence = 'Y increases as X increases.'
    if(k<0):
        sentence = 'Y decreases as X decreases.'
    return list(filter(None, sentence.split('.')))

def sentenceWrapper(sentence, focal=[],compare=[]):
    data = {}
    data['type'] = ''
    data['focus_id'] = focal
    data['compare_id'] = compare
    data['sentence'] = sentence
    data['sure'] = True
    return data

def arrayWrapper(data):
    arr = []
    d = data.tolist()
    for i in range(len(data)):
        datum = {}
        x, y = d[i]
        datum['q0'] = x
        datum['q1'] = y
        datum['id'] = i
        datum['c0'] = 0
        arr.append(datum)
    return arr

def json_wrapper(data, sentences):
    d = {}
    d['type'] = 'qq'
    d['color'] =  ["#bebada",  "#ccebc5",  "#d9d9d9",  "#bc80bd",  "#ffffb3",  "#fccde5",  "#fb8072",  "#80b1d3",  "#b3de69",  "#8dd3c7",  "#fdb462",  "#ffed6f"]
    d['vis_type'] = 'load_scatter_plot'
    d['major_name'] = 'q0'
    d['second_name'] = 'q1'
    d['sentences'] = [sentenceWrapper(sentence) for sentence in sentences]
    d['title'] = 'The Value'
    d['unit'] = ''
    d['q0'] = []
    d['q1'] = []
    d['data_array'] = arrayWrapper(data)
    d['unit1'] = 'X'
    d['unit2'] = 'Y'
    return d

def get_cluster_data(total_number = 25, centers = [[0.5, 0.5],[0.9,0.9]], corrs = [0.5, 0.9], outlierN = 0):
    # print('cluster debug', total_number, '\n', centers)
    clusterN = len(centers)
    total = total_number - clusterN * 3  # each cluster has at least 3 components
    amounts = []
    data = []
    maxR = max(max(centers))
    minR = min(min(centers))
    limit = maxR - minR
    for i in range(clusterN):
        number = random.randint(0,total)
        if (i==clusterN-1):
            number = total
        total -= number
        center = centers[i]
        k = random.random() / 5
        n = number+3
        amounts.append(n)
        c = 500 if n > 10 else 1000
        sigma =[k*np.sqrt(n)*limit*limit/c, np.sqrt(n)*k*limit*limit/c]
        corr = corrs[i]
        x = generate_class_data(center, n, sigma, corr)
        for d in x:
            data.append(d)
    data = numpy.asarray(data, dtype=np.int16)
    for i in range(outlierN):
        pass
    return data, amounts

def getRandomCenter(k, n, limit):
    r = random.randint(int(-limit/2/n), int(limit/2/n))
    x = max(0,k*limit + r) # uniform distribute over x axis
    y = random.randint(1,limit)
    return [x,y]

def generate_qq_data(data_amount=1000, datum_limit=(25,100), cluster_limit=5, printInfo=True):
    data = []
    for i in range(data_amount):
        r = random.randint(0,1)
        datumN = random.randint(datum_limit[0], datum_limit[1])
        datumN = max(cluster_limit*5, datumN) # data item number
        limit = random.randint(10,200)
        if(r==1): # cluster mode
            clusterN = random.randint(2,cluster_limit)
            centers = [getRandomCenter((i+1)/clusterN, clusterN, limit) for i in range(clusterN)]
            corrs = [random.random()/2+0.2 for i in range(clusterN)] #low correlation
            outlierN = random.randint(0,3) if random.randint(-1,1)==1 else 0
            datum, amounts = get_cluster_data(datumN, centers, corrs, outlierN)
            sentences = get_cluster_sentences(centers, amounts)
            data.append(json_wrapper(datum, sentences))
            if printInfo:
                print(amounts)
                print(sentences)
                visualize(datum)
        if(r==0): # line fitting mode
            r = [random.random()*limit for i in range(4)]
            start = [r[0], r[1]]
            end = [r[2], r[3]]
            sigm = random.random()/5 * abs(end[1]-start[1])**2 / 500
            datum = generate_line_data(start, end, sigm, datumN)
            sentences = get_line_sentences(start, end)
            if printInfo:
                print(sentences)
                visualize(datum)
            data.append(json_wrapper(datum, sentences))
    if(printInfo):
        print('\n The Json text are: \n',data)

    return data


def visualize(x):
    plt.scatter(x[:,0], x[:,1], marker = 'o', color = 'black', alpha = 0.2)
    plt.show()

def fileName(i):
    return f'scatter_{i:04}'

def saveFile(data, individual=True, folderName="./data", encoder=fileName):
    if individual:
        for i, d in enumerate(data):
            with open(f'{folderName}/{encoder(i)}.json', 'w') as outfile:
                json.dump(d, outfile, indent = 2)
    else:
        with open(f'all_data.json', 'w') as outfile:
                json.dump(data, outfile, indent = 2)


if __name__ == '__main__':
    data = generate_qq_data(1000, printInfo=False)
    saveFile(data, individual=True, folderName='./try_dir')
