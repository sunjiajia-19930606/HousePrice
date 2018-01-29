#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
正确率：.0.8258868128892499
时间：0:09:07.080122

2018-01-26 14:29:19.403704
0.675
2:06:07.178882
'''
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn import svm,preprocessing
from sklearn.externals import joblib
import math
import copy
import utils

def get_split_house(string):
    houseroomnum = 0
    houseparlournum = 0
    housetoiletnum = 0
    houseroom = re.match('(\d+)室', string)
    houseparlour = re.match('.*(\d+)厅', string)
    housetoilet = re.match('.*(\d+)卫', string)
    if houseroom:
        houseroomnum = int(houseroom.group(1))
    if houseparlour:
        houseparlournum = int(houseparlour.group(1))
    if housetoilet:
        housetoiletnum = int(housetoilet.group(1))
    return houseroomnum, houseparlournum, housetoiletnum


def get_tag(string):
    isnearmetro = 0
    isbathroom = 0
    iswindow = 0
    iskitchen = 0
    if string:
        if '地铁' in string:
            isnearmetro = 1
        if '独立卫生间' in string:
            isbathroom = 1
        if '独立阳台' in string:
            iswindow = 1
        if '厨房' in string:
            iskitchen = 1
    return isnearmetro, isbathroom, iswindow, iskitchen


def get_addition_in(string):
    bed = 0
    wardrobe = 0
    desk = 0
    airconditioning = 0
    table = 0
    heating = 0
    TV = 0
    gas = 0
    microwaveoven = 0
    electromagneticstove = 0
    waterheater = 0
    washingmachine = 0
    refrigerator = 0
    WIFI = 0
    sofa = 0
    cupboard = 0
    fumemachine = 0
    if string:
        if '床' in string:
            bed = 1
        if '衣柜' in string:
            wardrobe = 1
        if '书桌' in string:
            desk = 1
        if '空调' in string:
            airconditioning = 1
        if '餐桌' in string:
            table = 1
        if '暖气' in string:
            heating = 1
        if '电视机' in string:
            TV = 1
        if '燃气' in string:
            gas = 1
        if '微波炉' in string:
            microwaveoven = 1
        if '电磁炉' in string:
            electromagneticstove = 1
        if '热水器' in string:
            waterheater = 1
        if '洗衣机' in string:
            washingmachine = 1
        if '冰箱' in string:
            refrigerator = 1
        if 'WIFI' in string:
            WIFI = 1
        if '沙发' in string:
            sofa = 1
        if '橱柜' in string:
            cupboard = 1
        if '油烟机' in string:
            fumemachine = 1
    return bed, wardrobe, desk, airconditioning, table, heating, TV, gas, microwaveoven, electromagneticstove, \
           waterheater, washingmachine, refrigerator, WIFI, sofa, cupboard, fumemachine


# pd.DataFrame(train_X).to_csv('train_X.csv',encoding='gbk')
# pd.DataFrame(test_X).to_csv('test_X.csv',encoding='gbk')
@utils.show_time
def run():
    # bigdata_rent_count_shenzhen
    datas = pd.read_sql_query(
        "select * from bigdata_rent_count_shenzhen where source = '58' order by rent_time desc limit 100000",
        chunksize=50000, con=utils.engine,
        index_col='_id')
    # datas = pd.read_sql_table(table_name='bigdata_rent_count_shenzhen', chunksize=3000, con=utils.engine, index_col='_id')
    alldata = pd.DataFrame()
    for data in datas:
        newdata = copy.deepcopy(data)
        mydata = pd.DataFrame(index=newdata.index)

        mydata['rent_region'] = newdata['rent_region']
        mydata['rent_busi_area'] = newdata['rent_busi_area']

        mydata['rent_brand'] = [x.replace('(深圳）', '').replace('（深圳）', '') for x in newdata['rent_brand']]

        mydata['rent_lease_way'] = newdata['rent_lease_way']
        mydata['rent_floor'] = newdata['rent_floor'].astype('int')
        mydata['rent_area'] = newdata['rent_area']

        # 用价格/100作为标签
        mydata['rent_price'] = [round(float(x) / 10, 0) for x in newdata['rent_price']]

        median = round(mydata["rent_floor"].median(), 0)
        mydata["rent_floor"].fillna(median)

        tagdata = pd.DataFrame((get_tag(x) for x in newdata['rent_tags']), index=newdata.index,
                               columns=['isnearmetro', 'isbathroom', 'iswindow', 'iskitchen'])
        mydata = pd.concat((mydata, tagdata), axis=1)

        splitrentdata = pd.DataFrame((get_split_house(x) for x in newdata['rent_house_type']), index=newdata.index,
                                     columns=['rent_houseroom', 'rent_parlour', 'rent_toilet'])
        mydata = pd.concat((mydata, splitrentdata), axis=1)

        additionindata = pd.DataFrame((get_addition_in(x) for x in newdata['rent_addition_in']), index=newdata.index,
                                      columns=['bed', 'wardrobe', 'desk', 'airconditioning', 'table', 'heating', 'TV',
                                               'gas', 'microwaveoven', 'electromagneticstove', 'waterheater',
                                               'washingmachine', 'refrigerator', 'WIFI', 'sofa', 'cupboard',
                                               'fumemachine'])
        mydata = pd.concat((mydata, additionindata), axis=1)

        alldata = pd.concat((alldata, mydata), axis=0)

        # splitrentadditionin = pd.DataFrame((get_split_house(x) for x in newdata['rent_house_type']), index=newdata.index,
        # columns=['rent_houseroom','rent_parlour','rent_toilet'])

        # pd.DataFrame(newX).to_csv('newX.csv',encoding='gbk')
        # pd.DataFrame(newdata).to_sql('bigdata_rent_count_shenzhen1', if_exists='append', index=False, con=utils.engine)
    transfordata = pd.get_dummies(alldata)

    pd.DataFrame(transfordata).to_csv('transfordata.csv', encoding='gbk')
    target = transfordata['rent_price']
    #归一化
    # scaleddata = preprocessing.scale(transfordata.drop(columns='rent_price'))#暂时不做归一化 不方便之后测试
    scaleddata = transfordata.drop(columns='rent_price')
    # # 避免过拟合，采用交叉验证，验证集占训练集20%，固定随机种子（random_state)
    train_X, test_X, train_y, test_y = train_test_split(scaleddata,
                                                        target,
                                                        test_size=0.1,
                                                        random_state=1)

    clf = svm.SVC()
    clf.fit(train_X, train_y)
    joblib.dump(clf,'Train_SVM2.m')
    pre = clf.predict(test_X)
    prematri = np.array(pre)
    testmatri = np.array(test_y)
    resultacc = [math.fabs(x) for x in (prematri - testmatri)]
    acc = float(np.array([(i <= 30 and 1 or 0) for i in resultacc]).sum()) / len(resultacc)
    print(acc)


if __name__ == "__main__":
    run()