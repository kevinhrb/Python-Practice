import pandas as pd
import joblib


# 导入POD模型
def LoadPodModel(InputPath):
    Pod_model = joblib.load(InputPath)
    print("Model Loaded")
    return Pod_model


# 进行逆POD，返回对应的完整数据
def InversePod(Pod_model, NN_predict):
    result = Pod_model.inverse_transform(NN_predict)
    print("预测结果（逆pod下）：", result)
    return result


# 数据写入EXCEL, data为数据，Output为降维后数据保存路径
def Output(data, OutputPath):
    pdData = pd.DataFrame(data)
    writer = pd.ExcelWriter(OutputPath)  # 写入Excel文件
    pdData.to_excel(writer, '1')  # ‘1’是写入excel的sheet名
    writer.save()
    writer.close()
    print("Output Done")


# 以下需自行修改
# 神经网络预测值
SVM_predict = [1, 2, 3]
# POD模型存储位置
Pod_model = LoadPodModel(r'POD.model')
# 逆POD输出结果
result = InversePod(SVM_predict)
# 输出位置
OutputPath = r'SVM_predict_result.xlsx'
Output(result, OutputPath)
