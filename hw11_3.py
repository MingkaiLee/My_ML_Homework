import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from matplotlib.font_manager import *


model = MDS(dissimilarity='precomputed')


if __name__ == '__main__':
    # 读取数据
    city_name = pd.read_excel('./HomeworkData/city_dist.xlsx', usecols='A')
    city_name = np.array(city_name)
    distance_info = pd.read_excel('./HomeworkData/city_dist.xlsx', usecols='B:AI')
    distance_info = np.array(distance_info)
    # MDS及结果收集
    model.fit(distance_info)
    result = model.embedding_
    # 绘图
    plt.figure('MDS_city_location')
    plt.scatter(result[:, 0], result[:, 1])
    # 字体设置
    my_font = FontProperties(fname='./Library/Songti.ttc')
    for i in range(0, len(result)):
        plt.annotate(str(city_name[i]).strip('[\']'), xy=result[i], fontproperties=my_font)
    plt.show()
