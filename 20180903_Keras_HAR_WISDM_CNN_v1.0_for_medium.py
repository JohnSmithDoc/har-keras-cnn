# Compatibility layer between Python 2 and Python 3
# from __future__ import print_function
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn import preprocessing

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, GlobalAveragePooling1D
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
from keras.utils import np_utils

# %%

def feature_normalize(dataset):

    mu = np.mean(dataset, axis=0)
    sigma = np.std(dataset, axis=0)
    return (dataset - mu)/sigma


def show_confusion_matrix(validations, predictions):

    matrix = metrics.confusion_matrix(validations, predictions)
    plt.figure(figsize=(6, 4))
    sns.heatmap(matrix,
                cmap="coolwarm",
                linecolor='white',
                linewidths=1,
                xticklabels=LABELS,
                yticklabels=LABELS,
                annot=True,
                fmt="d")
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()


def show_basic_dataframe_info(dataframe,
                              preview_rows=20):

    """
    This function shows basic information for the given dataframe

    Args:
        dataframe: A Pandas DataFrame expected to contain data
        preview_rows: An integer value of how many rows to preview

    Returns:
        Nothing
    """

    # Shape and how many rows and columns
    print("Number of columns in the dataframe: %i" % (dataframe.shape[1]))
    print("Number of rows in the dataframe: %i\n" % (dataframe.shape[0]))
    print("First 20 rows of the dataframe:\n")
    # Show first 20 rows, preview_rows参数传递进来时已经默认为20了，预览前20行的数据
    print(dataframe.head(preview_rows))
    print("\nDescription of dataframe:\n")
    # Describe dataset like mean, min, max, etc.
    # print(dataframe.describe())


def read_data(file_path):

    """
    This function reads the accelerometer data from a file

    Args:
        file_path: URL pointing to the CSV file

    Returns:
        A pandas dataframe
    """

    column_names = ['user-id',
                    'activity',
                    'timestamp',
                    'x-axis',
                    'y-axis',
                    'z-axis']

    # 无表头文件需组合使用header=None和names参数，其实这里我们读的是txt，但其采集的格式和csv基本无差别，分隔符也一样
    df = pd.read_csv(file_path,
                     header=None,
                     names=column_names)
    # Last column has a ";" character which must be removed ...
    # 将z-axis列中所有出现的分号;替换为空字符串（即删除分号）
    # regex=True：启用正则表达式模式
    # inplace=True：直接修改原DataFrame，不返回新对象
    # to_replace=r';'：匹配目标字符（分号），r前缀表示原始字符串
    # value=r''：替换为的内容（空字符串）
    df['z-axis'].replace(regex=True,
      inplace=True,
      to_replace=r';',
      value=r'')

    # ... and then this column must be transformed to float explicitly
    df['z-axis'] = df['z-axis'].apply(convert_to_float)
    # This is very important otherwise the model will not fit and loss
    # will show up as NAN
    # 删除DataFrame中包含缺失值（NaN）的行
    # axis=0：沿行轴操作（删除行）
    # how='any'：只要行中有任意列存在缺失值就删除该行
    # inplace=True：直接在原DataFrame上修改，不返回新对象
    df.dropna(axis=0, how='any', inplace=True)

    return df


def convert_to_float(x):

    try:
        return np.float(x)
    except:
        return np.nan


# Not used right now
def feature_normalize(dataset):

    mu = np.mean(dataset, axis=0)
    sigma = np.std(dataset, axis=0)
    return (dataset - mu)/sigma


def plot_axis(ax, x, y, title):
    # 注意这个ax也是传递进来的，说明更大的画图函数会调用这个plot_axis，这个函数在本文件就是plot_activity
    # 绘制x y 曲线
    ax.plot(x, y)
    ax.set_title(title)
    # x轴刻度隐藏
    ax.xaxis.set_visible(False)
    # 自动计算y 轴显示范围
    ax.set_ylim([min(y) - np.std(y), max(y) + np.std(y)])
    ax.set_xlim([min(x), max(x)])
    # 启用网格线辅助度数
    ax.grid(True)


def plot_activity(activity, data):

    # 使用plt.subplots创建3行1列的垂直子图布局（x/y/z三轴数据分开显示），设置15x10英寸的画布尺寸，且x轴刻度自动对齐
    # sharex=True 所有子图共享x轴刻度
    # fig：创建整个画布对象
    # (ax0, ax1, ax2)：解包获得三个子图Axes对象的元组
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3,
         figsize=(15, 10),
         sharex=True)

    # 子图的传入参数，横轴数据，纵轴数据，子图标题
    plot_axis(ax0, data['timestamp'], data['x-axis'], 'x-axis')
    plot_axis(ax1, data['timestamp'], data['y-axis'], 'y-axis')
    plot_axis(ax2, data['timestamp'], data['z-axis'], 'z-axis')
    # hspace=0.2控制子图垂直间距为20%
    # suptitle将活动类型名称作为总标题显示在顶部
    # top=0.90保留10%的顶部边距防止标题重叠
    plt.subplots_adjust(hspace=0.2)
    fig.suptitle(activity)
    plt.subplots_adjust(top=0.90)
    plt.show()



def create_segments_and_labels(df, time_steps, step, label_name):

    """
    This function receives a dataframe and returns the reshaped segments
    of x,y,z acceleration as well as the corresponding labels

    Args:
        df: Dataframe in the expected format
        time_steps: Integer value of the length of a segment that is created
    Returns:
        reshaped_segments
        labels:
    """

    # x, y, z acceleration as features
    N_FEATURES = 3
    # Number of steps to advance in each iteration (for me, it should always
    # be equal to the time_steps in order to have no overlap between segments)
    # step = time_steps
    segments = []
    labels = []

    # 按时间片标记lable
    # 在时间序列数据上创建固定长度的滑动窗口
    # 0：起始索引位置
    # len(df) - time_steps：确保窗口不超过数据边界
    # step：控制窗口移动步长（可跳过部分数据）
    # range(start, stop, step)按步长step生成序列
    for i in range(0, len(df) - time_steps, step):
        # .values 将Pandas Series转换为NumPy数组格式，提升后续计算效率,然后取连续一时间段的数据，time_steps的时间段
        xs = df['x-axis'].values[i: i + time_steps]
        ys = df['y-axis'].values[i: i + time_steps]
        zs = df['z-axis'].values[i: i + time_steps]
        # Retrieve the most often used label in this segment
        # stats.mode() 使用scipy.stats模块的mode函数计算该数据段中出现频率最高的标签值
        # 其返回值为包含两个元素的元组：第一个元素是众数值数组,第二个元素是出现次数数组
        # 第一个[0]获取众数值数组
        # 第二个[0]提取众数值数组中的第一个众数（当存在多个并列众数时取第一个)
        # 参数keepdims=True来避免兼容性警告
        label = stats.mode(df[label_name][i: i + time_steps], keepdims=True)[0][0]
        segments.append([xs, ys, zs])
        labels.append(label)

    # Bring the segments into a better shape
    # 将传感器时间序列数据转换为深度学习模型所需的3D输入格式，原始列表转化为32位浮点型NumPy数组
    # 第一维-1：自动计算样本数量（batch_size）
    # 第二维time_steps：每个样本的时间步长（窗口长度）
    # 第三维N_FEATURES：特征维度（通常对应x/y/z三轴）
    # 这里得到的 reshaped_segments 其中和一个label对应的内容如下
    #  [[ 0.142857  0.154532  0.279062]
    #   [ 0.475587  0.300466  0.154532]
    #   [ 0.181773  0.055296 -0.053668]
    #   ...
    #   [-0.956783 -0.321844 -0.685876]
    #   [-0.234364  0.011146 -0.931386]
    #   [-0.225898 -0.335954  1.012938]]
    # TensorFlow/Keras‌：1D-CNN输入形状通常为(batch_size, sequence_length, channels)
    reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(-1, time_steps, N_FEATURES)
    labels = np.asarray(labels)

    # 最后segments 和 label 一一匹配
    return reshaped_segments, labels

# %%

# ------- THE PROGRAM TO LOAD DATA AND TRAIN THE MODEL -------
# 整个程序从这里开始运行，实际应当有一个main会更好一些 --------------------------------------------------------------------------

# Set some standard parameters upfront
# 设置Pandas输出浮点数时保留1位小数，统一数据展示格式
pd.options.display.float_format = '{:.1f}'.format
# 启用Seaborn的默认主题和配色方案
sns.set() # Default seaborn look and feel
# 指定Matplotlib使用ggplot风格的绘图主题（灰色背景、高对比度颜色等）
plt.style.use('ggplot')
print('keras version ', keras.__version__)

# %%

print("\n--- Load, inspect and transform data ---\n")

# Load data set containing all the data from csv
df = read_data('WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt')

# Describe the data
# 显示/预览前20行数据
show_basic_dataframe_info(df, 20)

# df['activity'].value_counts() 对DataFrame中'activity'列的值进行频数统计，返回每个活动类型出现的次数
# .plot(kind='bar') 将统计结果用柱状图展示
# 设置图标标题为 Training Examples by Activity Type
df['activity'].value_counts().plot(kind='bar',
                                   title='Training Examples by Activity Type')
plt.show()
# plt.pause(0.1)  # 短暂暂停，确保图像渲染完成

df['user-id'].value_counts().plot(kind='bar',
                                  title='Training Examples by User')
plt.show()
# plt.pause(0.1)  # 短暂暂停，确保图像渲染完成

# 取数据框中所有不重复的活动类型，即np.unique(df["activity"])得到的是
# ['Downstairs' 'Jogging' 'Sitting' 'Standing' 'Upstairs' 'Walking']
# print(np.unique(df["activity"]))

for activity in np.unique(df["activity"]):
    # 比如activity是Jogging，则df[df["activity"] == activity]就选出了所有Jogging活动的内容，然后放在了df中
    # 接着取 activity为Jogging的df的前180行作为子集
    # df[df["activity"] == activity]返回的是过滤后（行过滤）的DataFrame子集‌
    # 然后我们取前180个数据
    subset = df[df["activity"] == activity][:180]
    # 将这个活动的子集画出来，这里是180个数据，大概9s的数据
    plot_activity(activity, subset)

# Define column name of the label vector
LABEL = "ActivityEncoded"
# Transform the labels from String to Integer via LabelEncoder
# 这里进行了数字编码，看一下后面有没有再处理？因为实际应该需要使用one-hot编码更好
le = preprocessing.LabelEncoder()
# Add a new column to the existing DataFrame with the encoded values
df[LABEL] = le.fit_transform(df["activity"].values.ravel())
# %%

print("\n--- Reshape the data into segments ---\n")

# Differentiate between test set and training set
# 根据用户id进行划分，这里好神奇
# 这里默认使用深拷贝，避免警告，但是可能内存消耗大
df_test = df[df['user-id'] > 28].copy()
df_train = df[df['user-id'] <= 28].copy()

# Normalize features for training data set
# 三个特征数据均正则化
# 这里应该使用 df_train进行正则化吧，不然整个df还是原先的数据啊，这里是不是原始代码有问题？
# df_train['x-axis'] = feature_normalize(df['x-axis'])
# df_train['y-axis'] = feature_normalize(df['y-axis'])
# df_train['z-axis'] = feature_normalize(df['z-axis'])

# 我修改为如下：
df_train['x-axis'] = feature_normalize(df_train['x-axis'])
df_train['y-axis'] = feature_normalize(df_train['y-axis'])
df_train['z-axis'] = feature_normalize(df_train['z-axis'])
# Round in order to comply to NSNumber from iOS
# 进行六位小数保留
df_train = df_train.round({'x-axis': 6, 'y-axis': 6, 'z-axis': 6})

LABELS = ["Downstairs",
          "Jogging",
          "Sitting",
          "Standing",
          "Upstairs",
          "Walking"]

# 时间段的补偿，20Hz的采样率，80个时间段表明是4秒
# The number of steps within one time segment
TIME_PERIODS = 80
# The steps to take from one segment to the next; if this value is equal to
# TIME_PERIODS, then there is no overlap between the segments
# 每个时间片的距离，这里是40表明有50%的重叠率
STEP_DISTANCE = 40
# Reshape the training data into segments
# so that they can be processed by the network
x_train, y_train = create_segments_and_labels(df_train,
                                              TIME_PERIODS,
                                              STEP_DISTANCE,
                                              LABEL)

# %%

print("\n--- Reshape data to be accepted by Keras ---\n")

# Inspect x data
print('x_train shape: ', x_train.shape)
# Displays (20869, 40, 3)
print(x_train.shape[0], 'training samples')
# Displays 20869 train samples

# Inspect y data
print('y_train shape: ', y_train.shape)
# Displays (20869,)

# Set input & output dimensions
# 时间周期，即时间戳个数， 传感器个数
num_time_periods, num_sensors = x_train.shape[1], x_train.shape[2]

# 这里打印了之前labelEncoder处理时得到的分类
num_classes = le.classes_.size
print(list(le.classes_))


# 以下这些是考虑coreML兼容性涉及的，丢掉了维度信息，为了能在ios上运行，但我们不考虑这个，就直接使用keras
# Set input_shape / reshape for Keras
# Remark: acceleration data is concatenated in one array in order to feed
# it properly into coreml later, the preferred matrix of shape [40,3]
# cannot be read in with the current version of coreml (see also reshape
# layer as the first layer in the keras model)
# input_shape = (num_time_periods*num_sensors)
# x_train = x_train.reshape(x_train.shape[0], input_shape)
#
# print('x_train shape:', x_train.shape)
# # x_train shape: (20869, 120)
# print('input_shape:', input_shape)
# # input_shape: (120)
# 以上这些是考虑coreML兼容性涉及的，丢掉了维度信息，为了能在ios上运行，但我们不考虑这个，就直接使用keras-----------------------


# Convert type for Keras otherwise Keras cannot process the data
# 将训练特征数据转换为单精度浮点数，标签数据也是
x_train = x_train.astype("float32")
y_train = y_train.astype("float32")

# %%

# One-hot encoding of y_train labels (only execute once!)
# np_utils.to_categorical() 是Keras的工具函数，用于将整型类别标签（如[1,3,0]）转换为二进制矩阵形式的独热编码（如[[0,1,0], [0,0,1], [1,0,0]]）
# num_classes 参数指定总类别数，决定输出矩阵的列数
y_train = np_utils.to_categorical(y_train, num_classes)
print('New y_train shape: ', y_train.shape)
# (4173, 6)

# %%

print("\n--- Create neural network model ---\n")

# 1D CNN neural network
model_m = Sequential()
# model_m.add(Reshape((TIME_PERIODS, num_sensors), input_shape=(input_shape,)))
model_m.add(Conv1D(100, 10, activation='relu'))
model_m.add(Conv1D(100, 10, activation='relu'))
model_m.add(MaxPooling1D(3))
model_m.add(Conv1D(160, 10, activation='relu'))
model_m.add(Conv1D(160, 10, activation='relu'))
model_m.add(GlobalAveragePooling1D())
model_m.add(Dropout(0.5))
model_m.add(Dense(num_classes, activation='softmax'))
# print(model_m.summary())
# Accuracy on training data: 99%
# Accuracy on test data: 91%
# 实际验证下来训练集精度可达99%以上，但是验证集准确率只有82%左右
# 但是在test数据集上可以到90%

# %%

print("\n--- Fit the model ---\n")

# The EarlyStopping callback monitors training accuracy:
# if it fails to improve for two consecutive epochs,
# training stops early
callbacks_list = [
    keras.callbacks.ModelCheckpoint(
        filepath='model_params/best_model.{epoch:02d}-{val_loss:.2f}.h5',
        monitor='val_loss', save_best_only=True),
    # 监控训练集准确率（accuracy），若连续4个周期（patience=1）未提升则提前终止训练
    keras.callbacks.EarlyStopping(monitor='accuracy', patience=4)
]

model_m.compile(loss='categorical_crossentropy',
                optimizer='adam', metrics=['accuracy'])

# 修改下训练方式，学习率为0.01
# model_m.compile(loss='categorical_crossentropy',
#                 optimizer=keras.optimizers.Adam(0.1), metrics=['accuracy'])

# Hyper-parameters
BATCH_SIZE = 400
EPOCHS = 50

# Enable validation to use ModelCheckpoint and EarlyStopping callbacks.
history = model_m.fit(x_train,
                      y_train,
                      batch_size=BATCH_SIZE,
                      epochs=EPOCHS,
                      callbacks=callbacks_list,
                      validation_split=0.2,
                      verbose=1)

# %%

print("\n--- Learning curve of model training ---\n")

# summarize history for accuracy and loss
plt.figure(figsize=(6, 4))
plt.plot(history.history['accuracy'], "g--", label="Accuracy of training data")
plt.plot(history.history['val_accuracy'], "g", label="Accuracy of validation data")
plt.plot(history.history['loss'], "r--", label="Loss of training data")
plt.plot(history.history['val_loss'], "r", label="Loss of validation data")
plt.title('Model Accuracy and Loss')
plt.ylabel('Accuracy and Loss')
plt.xlabel('Training Epoch')
plt.ylim(0)
plt.legend()
plt.show()

#%%

print("\n--- Check against test data ---\n")

# Normalize features for training data set
df_test['x-axis'] = feature_normalize(df_test['x-axis'])
df_test['y-axis'] = feature_normalize(df_test['y-axis'])
df_test['z-axis'] = feature_normalize(df_test['z-axis'])

df_test = df_test.round({'x-axis': 6, 'y-axis': 6, 'z-axis': 6})

x_test, y_test = create_segments_and_labels(df_test,
                                            TIME_PERIODS,
                                            STEP_DISTANCE,
                                            LABEL)

# Set input_shape / reshape for Keras
# 这里无需再进行reshape
# x_test = x_test.reshape(x_test.shape[0], input_shape)

x_test = x_test.astype("float32")
y_test = y_test.astype("float32")

y_test = np_utils.to_categorical(y_test, num_classes)

score = model_m.evaluate(x_test, y_test, verbose=1)

print("\nAccuracy on test data: %0.2f" % score[1])
print("\nLoss on test data: %0.2f" % score[0])

# %%

print("\n--- Confusion matrix for test data ---\n")

y_pred_test = model_m.predict(x_test)
# Take the class with the highest probability from the test predictions
max_y_pred_test = np.argmax(y_pred_test, axis=1)
max_y_test = np.argmax(y_test, axis=1)

show_confusion_matrix(max_y_test, max_y_pred_test)

# %%

print("\n--- Classification report for test data ---\n")

print(classification_report(max_y_test, max_y_pred_test))