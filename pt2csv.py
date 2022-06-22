import torch
import csv
import os

# path1 = "/NAS01/TCGA_BRCA/processed/feat-x20-RN50-B-color_norm/pt_files/"
# ptnames = os.listdir(path1)
# # pt_f = torch.load(path1+ptnames[0])
# idx=1
# for ptfile in ptnames:
#     print("write", idx)
#     idx += 1
    
#     dir = "/NAS01/TCGA_BRCA/processed/feat-x20-RN50-B-color_norm/csv_files/"
#     csv_file = "-".join(ptfile.split("-")[:3])
#     with open(dir + csv_file + ".csv", "w", encoding="utf-8") as f:
#         csv_writer = csv.writer(f)
#         pt_f = torch.load(path1+ptfile)
#         pt_f=pt_f.cpu().numpy()
#         csv_writer.writerows(pt_f)

with open("csvlabel.csv", "r", encoding="utf-8") as f:
    data = csv.reader(f)
    data = list(data)

for line in data:
    line[0]="/NAS01/TCGA_BRCA/processed/feat-x20-RN50-B-color_norm/csv_files/" + line[0][7:]

with open("csvlabel.csv", "w", encoding="utf-8") as f:
    csv_writer = csv.writer(f)
    csv_writer.writerows(data)