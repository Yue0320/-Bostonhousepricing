#引入库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#开始建立线性回归模型
class linear_regression():#建立一个线性模型的类
    def fitness(self,Date_X_input,Date_Y,learning_rate=0.5,lamda=0.03):#一个函数（拟合函数），输入有5个，self，X，Y，leaningrate n步长（每次优化夺少），lamda（截距？）
        sample_num,property_num=Date_X_input.shape#np.shape读取矩阵的长度  样本个数，样本属性个数
        Date_X = np.c_[Date_X_input, np.ones(sample_num)]
        self.theta = np.zeros([property_num + 1, 1]) #np.zeros：返回来一个给定形状和类型的用0填充的数组；theta θ一个矩阵参数
        Max_count=int(1e8)#最多迭代次数
        last_better = 0  # 上一次得到较好学习误差的迭代学习次数
        last_Jerr = int(1e8)  # 上一次得到较好学习误差的误差函数值
        threshold_value = 1e-8  # 定义在得到较好学习误差之后截止学习的阈值   误差范围
        threshold_count = 10  # 定义在得到较好学习误差之后截止学习之前的学习次数
        for step in range(0,Max_count):#开始循环迭代
            predict=Date_X.dot(self.theta)#预测  numpy.dot：处理数组/矩阵，计算点积（矩阵相乘）
            J_theta=sum((predict-Date_Y)**2/2*sample_num)#损失函数
            self.theta -= learning_rate * (lamda * self.theta + (Date_X.T.dot(predict - Date_Y)) / sample_num) #更新theta，把损失的加上
            if J_theta < last_Jerr - threshold_value:# 检测损失函数的变化值，满足条件提前结束迭代
                   last_Jerr = J_theta#更新
                   last_better = step#更新
            elif step - last_better > threshold_count:
                break
            if step % 50 == 0:# 定期打印，方便用户观察变化
                print("step %s: %.6f" % (step, J_theta))
    def predicted(self,X_input):#定义一个预测函数
        sample_num = X_input.shape[0]#np.shape读取矩阵的长度  个数只读一行
        X = np.c_[X_input, np.ones(sample_num, )]#np.c 连接两个矩阵
        predict = X.dot(self.theta)
        return predict
def property_label(pd_data):#属性标签  对数据集中的样本属性进行分割，制作X和Y矩阵
    row_num = pd_data.shape[0]
    column_num = len(pd_data.iloc[0, 0].split())# 行数、列数
    X = np.empty([row_num, column_num - 1])
    Y = np.empty([row_num, 1])#初始化 空数组
    for i in range(0, row_num):
        row_array = pd_data.iloc[i, 0].split()#np.iloc取行数列数的函数  np.split 切分
        X[i] = np.array(row_array[0:-1])
        Y[i] = np.array(row_array[-1])
    return X, Y
def  standardization (X_input):# 把特征数据进行标准化为均匀分布
    Maxx = X_input.max(axis=0)
    Minx = X_input.min(axis=0)
    X = (X_input - Minx) / (Maxx - Minx)
    return X, Maxx, Minx
if __name__ == "__main__":#文件本身作为脚本直接执行
    data = pd.read_csv("housing-data.csv", header=None)#pandas的读取scv文件
    Date_X, Date_Y = property_label(data)    # 对训练集进行X，Y分离
    Standard_DateX, Maxx, Minx =  standardization (Date_X)    # 对X进行归一化处理，方便后续操作
    
    model = linear_regression()
    model.fitness(Standard_DateX, Date_Y)
    
    Date_predict = model.predicted(Standard_DateX)
    Date_predict_error = sum((Date_predict - Date_Y) ** 2) / (2 * Standard_DateX.shape[0])
    print("Test error is %d" % (Date_predict_error))
    print(model.theta)
    t = np.arange(len(Date_predict))
    #绘图
    plt.figure(facecolor='white')#背景颜色
    plt.plot(t, Date_Y, '#00AAAA', lw=1,label=u'actual value')#绘制Y的线 真实价格
    plt.plot(t, Date_predict, '#FF5555', lw=1.6, label=u'estimated value')#绘制预测结果    #青衣&朱鸢配色
    plt.legend(loc='upper right')
    plt.title(u'Boston house price', fontsize=20)
    plt.xlabel(u'case id', fontsize=12)
    plt.ylabel(u'house price', fontsize=12)
    plt.grid()#添加网格线
    plt.show()
