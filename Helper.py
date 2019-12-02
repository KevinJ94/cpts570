from keras_preprocessing.image import ImageDataGenerator
import numpy as np


def read_labels(path):
    import  csv
    csvFile = open(path, "r")
    reader = csv.reader(csvFile)
    result = {}
    for item in reader:
        if reader.line_num == 1:
            continue
        result[item[0]] = item[1]
    csvFile.close()
    return result


#Image Preprocess
def image_process():
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=45,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.25,
        horizontal_flip=True,
        fill_mode='nearest')

    training_set = train_datagen.flow_from_directory(
        'data/train_x',
        target_size=(400, 400),
        batch_size=32,
        seed=10)

    test_datagen = ImageDataGenerator(rescale=1./255)

    test_set = test_datagen.flow_from_directory(
        'data/test_x',
        target_size=(400, 400),
        batch_size=32,
        seed=10)

    return training_set,test_set

def plot(model,title):
    y_axis = model.history["accuracy"]
    x_axis = list(range(1, len(y_axis) + 1))
    save_curve(x_axis=x_axis, y_axis=y_axis, title=title+'training', xlabel="epoch", ylabel="accuracy")
    y_axis = model.history["val_accuracy"]
    save_curve(x_axis=x_axis, y_axis=y_axis, title=title+'testing', xlabel="epoch", ylabel="accuracy")
    y_axis = model.history["val_loss"]
    save_curve(x_axis=x_axis, y_axis=y_axis, title=title + 'val_loss', xlabel="epoch", ylabel="accuracy")
    y_axis = model.history["loss"]
    save_curve(x_axis=x_axis, y_axis=y_axis, title=title + 'loss', xlabel="epoch", ylabel="accuracy")

def save_curve(x_axis, y_axis, title, xlabel, ylabel):
    import matplotlib.pyplot as plt
    import os
    x_axis = np.array(x_axis)
    y_axis = np.array(y_axis)
    plt.plot(x_axis, y_axis)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if not os.path.exists("figures/"):
        os.makedirs("figures/")
    plt.savefig("figures/" + title)
    plt.clf()
    plt.close()

# acc_list = []
# val_acc_list = []
# f = open("data/res.txt")
# for line in f.readlines():
#     temp = line[10:-1]
#     temp = temp.split('-')
#     acc =temp[1][6:-1]
#     val_acc = temp[3][10:]
#     acc_list.append(float(acc))
#     val_acc_list.append(float(val_acc))
#
# y_axis = acc_list
# x_axis = list(range(1, len(y_axis) + 1))
# save_curve(x_axis=x_axis, y_axis=y_axis, title="InceptionV3_training_accuracy_" + str(50),
#                   xlabel="epoch",
#                   ylabel="accuracy")
# y_axis = val_acc_list
# save_curve(x_axis=x_axis, y_axis=y_axis, title="InceptionV3_testing_accuracy_" + str(50), xlabel="epoch",
#                   ylabel="accuracy")
#

# def rename():
#     import  os
#     dirs = os.listdir("../Images")
#     for dir in dirs:
#         print('oldname is:' + dir)  # 输出老的名字
#         new_name = dir[10:].lower()
#         print(new_name)
#
#         oldname = os.path.join("../Images", dir)  # 老文件夹的名字
#         newname = os.path.join("../Images", new_name)  # 新文件夹的名字
#
#         os.rename(oldname, newname)


# DO NOT run this function !!!
# def generate_test(folder_path):
#     import os
#     import random
#     import shutil
#
#     file_list = os.listdir(folder_path)
#     dic = read_labels("data/labels.csv")
#     count_dic = {}
#     val_list = set(dic.values())
#     val_list = sorted(val_list)
#     for val in val_list:
#         t = []
#         for k, v in dic.items():
#             if v == val:
#                 t.append(k)
#         count_dic[val] = t
#     # print(count_dic)
#     test_list = []
#
#     #train_dic = {}
#     test_dic = {}
#
#     for k, v in count_dic.items():
#         ran_list = random.sample(v, int(len(v)*0.2))
#         test_list.extend(ran_list)
#
#     for k, v in count_dic.items():
#         test = []
#         train = []
#         for i in v:
#             if i in test_list:
#                 test.append(i)
#             else:
#                 train.append(i)
#         #train_dic[k] = train
#         test_dic[k] = test
#
#     # for k, v in train_dic.items():
#     #     folder = os.path.exists("data/train_x/"+k)
#     #     if not folder:
#     #         os.makedirs("data/train_x/"+k)
#     #     for i in v:
#     #         s = "data/train/" + i + '.jpg'
#     #         d = "data/train_x/" + k + '/' + i + '.jpg'
#     #         shutil.copy(s,d)
#
#     for k, v in test_dic.items():
#         folder = os.path.exists("data/test_x/"+k)
#         if not folder:
#             os.makedirs("data/test_x/"+k)
#         for i in v:
#             s = "data/train/" + i + '.jpg'
#             d = "data/test_x/" + k + '/' + i + '.jpg'
#             shutil.copy(s,d)
#
# generate_test("data/train")


