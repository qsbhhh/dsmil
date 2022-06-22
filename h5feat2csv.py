import h5py
import os




# path = "lbp_feat/"

# h5names = os.listdir(path)

# print(len(h5names))



# # 安装h5py库,导入
# f = h5py.File(path + h5names[0], 'r')
# for key in f.keys():
#     print(f[key].name)
#     print(f[key].shape)
#     print(f[key][:])
#     # # 将key换成某个文件中的关键字,打印出该数据(数组)





import csv
# import numpy as np

# with open("test.csv", "w", encoding="utf-8") as f:
#     csv_writer = csv.writer(f)
#     head  = [str(i) for i in range(256)]
#     head = [" "] + head

#     csv_writer.writerow(head)


#     h5_f =  h5py.File(path + h5names[0], 'r')
#     features = h5_f['lbp_feature']
#     # print(type(features[0]))
#     for i in range(len(features)):
#         csv_writer.writerow(np.append([i+1], features[i]))
    

    


# print("-".join(h5names[0].split("-")[:3]))


#把h5文件转化成csv
# idx = 1
# for h5file in h5names:
#     print("writer ", idx)
#     idx += 1

#     dir = "lbpcsv/"
#     f_name = "-".join(h5file.split("-")[:3])
    
#     with open(dir + f_name + ".csv", "w", encoding="utf-8") as f:
#         csv_writer = csv.writer(f)
#         # head  = [str(i) for i in range(256)]
#         # head = [" "] + head

#         # csv_writer.writerow(head)

#         h5_f =  h5py.File(path + h5file, 'r')
#         features = h5_f['lbp_feature']
#         # print(type(features[0]))
#         for i in range(len(features)):
#             csv_writer.writerow(features[i])
   
import csv

path = "lbpcsv/"

csvnames = os.listdir(path)

print(len(csvnames))
idx = 1
for csvname in csvnames:
    print(idx)
    idx += 1
    
    data = []
    with open(path + csvname, "r", encoding="utf-8") as f:
        data = csv.reader(f)
        data = list(data)
    

    for i in range(len(data)):
        data[i]=data[i]*2
        print(len(data[i]))
           
    with open(path + csvname, "w", encoding="utf-8") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerows(data)
    

# data = ""
# with open("csvlabel.csv", "r", encoding="utf-8") as f:
#     data = csv.reader(f)
#     data = list(data)

# # print(data[0])


# filtered_data = []
# name_set = set()

# for line in data:
#     line[1] = line[1] + ".csv"
#     if(line[1] in csvnames):
#         if(line[1] in name_set):
#             continue

#         name_set.add(line[1])
#         line[1] = "lbpcsv/" + line[1]
#         filtered_data.append(line)




# print(len(filtered_data))

# for i,  line in enumerate(filtered_data, 1):
#     line[0] = i
# # print(filtered_data[0:5])

# with open("csvlabel.csv", "w", encoding="utf-8") as f:
#     csv_writer = csv.writer(f)
#     csv_writer.writerow(data[0])
#     csv_writer.writerows(filtered_data)
# print("over")