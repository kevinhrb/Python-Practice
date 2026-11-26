import xlrd
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import joblib


# 打开数据文件，读入xls格式数据，InputPath为样本数据路径，读入的数据维度为x*y,其中x为一个样本的节点数，y为总样本个数
def ReadSampleData(InputPath, x, y):
    workbook = xlrd.open_workbook(InputPath)
    sheet1 = workbook.sheets()[0]
    # 每一行为一个维度，将每一个维度的数据提取出来
    data = np.zeros(shape=(0, y))
    for i in range(0, x):
        rows = sheet1.row_values(i)
        data = np.append(data, [rows], axis=0)
    print(data)
    return data


# 进行数据降维，data为导入的样本数据, n为降维后的数据维度
def POD(data, n):

    data = data.astype(float)  # 转换data内数据由string类型至float类型
    cov_f = np.cov(data)
    print("协方差矩阵：", cov_f)

    eigenvalue, featurevector = np.linalg.eig(cov_f)
    print("特征值：", eigenvalue)
    print("特征向量：", featurevector)

    data = data.T
    pca = PCA(n_components=n)
    pca.fit(data)
    data_afterPOD = pca.transform(data)
    print("降维后的数据：", data_afterPOD)

    # 降维后的各主成分的方差值占总方差值的比例，即方差贡献率
    print("方差贡献率：", pca.explained_variance_ratio_)

    # 总方差贡献率
    print("总方差贡献率", sum(pca.explained_variance_ratio_))

    # 降维后的各主成分的方差值
    print("主成分方差：", pca.explained_variance_)
    return pca, data_afterPOD  # pca为降维模型，data_afterPOD为降维后的数据


# 降维后的数据写入EXCEL, data_afterPOD为降维后的数据，OutputPath为降维后数据保存路径
def Output(data_afterPOD, OutputPath):
    pdData = pd.DataFrame(data_afterPOD)
    writer = pd.ExcelWriter(OutputPath)  # 写入Excel文件
    pdData.to_excel(writer, '1')  # ‘1’是写入excel的sheet名
    writer.save()
    writer.close()
    print("Output Done")


# 保存POD模型
def SavePodModel(Pod_model, OutputPath):
    joblib.dump(Pod_model, OutputPath)
    print("Model Saved")


# POD程序运行,需要自行修改以下代码
SampleDataInputPath = r'example.xlsx'
data = ReadSampleData(SampleDataInputPath, 150, 50)
pca, data_afterPOD = POD(data, 3)
OutputPath1 = r'example_afterPOD.xlsx'
Output(data_afterPOD, OutputPath1)
OutputPath2 = r'POD.model'
SavePodModel(pca, OutputPath2)