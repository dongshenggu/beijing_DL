import os
import SimpleITK as sitk
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import numpy as np
import shutil
# import itertools

# series = ['A', 'N', 'V']
# file_cates = ['ori', 'seg']
# dataPath = r'H:\Wei\liaoxiao\Process\Normal_Data'
# patients = os.listdir(dataPath)
# for patient in patients:
#     print(patient)
#     for serie, file_cate in itertools.product(series, file_cates):
#         performPath = os.path.join(dataPath, patient, serie, file_cate)
#         files = os.listdir(performPath)
#         os.remove(os.path.join(performPath, files[0]))
#         os.remove(os.path.join(performPath, files[-1]))

# data_path = r'H:\Wei\liaoxiao\Process\Normal_Data'
# clinical_info_path = r'H:\Wei\liaoxiao\clinical.xlsx'
# datafiles = os.listdir(data_path)
# clinical_info = pd.read_excel(clinical_info_path, header=0)
# pd.DataFrame(list(zip(datafiles, clinical_info['影像号'].tolist())), columns=('data', 'clinical')).to_csv(r'C:\Users\GDS\Desktop\tmp.csv',index=False)

# clinical_info_path = r'H:\Wei\liaoxiao\clinical.xlsx'
# clinical_info = pd.read_excel(clinical_info_path, header=0)
# new_cli = clinical_info[['影像号', '诊断时间', 'REICST']]
# new_cli.sort_values(axis=0, ascending=True, by='诊断时间',inplace=True)
# training_index = int((2/3)*(new_cli.shape[0]))
# training = new_cli.iloc[:training_index, ]
# test = new_cli.iloc[training_index:, ]
# oriFolder = r'H:\Wei\liaoxiao\Process\Normal_Data'
# training_fold = r'H:\Wei\liaoxiao\Process\training'
# test_fold = r'H:\Wei\liaoxiao\Process\test'
# pd.DataFrame(training).to_csv(r'C:\Users\GDS\Desktop\tmp1.csv')
#
'''divide into training and validation'''
# names = training['影像号'].tolist()
# label = training['REICST'].tolist()
# for patient, label in zip(names, label):
#     ori_file = os.path.join(oriFolder, str(patient))
#     copy_file = os.path.join(training_fold, str(label), str(patient))
#     shutil.copytree(ori_file, copy_file)
# names = test['影像号'].tolist()
# label = test['REICST'].tolist()
# for patient, label in zip(names, label):
#     ori_file = os.path.join(oriFolder, str(patient))
#     copy_file = os.path.join(test_fold, str(label), str(patient))
#     shutil.copytree(ori_file, copy_file)

# path1 = r'H:\Wei\liaoxiao\Process\Beijing_image_info1.csv'
# path2 = r'H:\Wei\liaoxiao\Process\Beijing_image_info2.csv'
# info1 = pd.read_csv(path1, encoding='gbk', header=0)
# patientName = info1.name.tolist()
# index_bool = ['V' in patient for patient in patientName]
# index1 = np.where(np.array(index_bool)==True)[0].tolist()
#
# info2 = pd.read_csv(path2, encoding='gbk', header=0)
# patientName = info2.name.tolist()
# index_bool = ['V' in patient for patient in patientName]
# index2 = np.where(np.array(index_bool)==True)[0].tolist()
#
# pd.concat([info1.iloc[index1, :], info2.iloc[index2, :]], axis=0).to_csv(r'H:\Wei\liaoxiao\Process\Beijing_image_infoV.csv', index=False)

'''shape to three columns'''
# series = ['A', 'N', 'V']
# for serie in series:
#     path = 'H:\Wei\liaoxiao\Process\Beijing_image_info' + serie + '.csv'
#     serie_info = pd.read_csv(path, encoding='gbk', header=0)
#     shape = serie_info['shape']
#     new_shape = [eval(member) for member in shape]
#     ranges = [newmember[0]-2 for newmember in new_shape]
#     serie_info['range'] = ranges
#     serie_info.to_csv(path, index=False)

'add time and label info to Beijing_image_info excel'
# cli_path = r'H:\Wei\liaoxiao\clinical.xlsx'
# info_path = r'H:\Wei\liaoxiao\Process\Beijing_image_infoV.csv'
# cli_info = pd.read_excel(cli_path, header=0)
# info = pd.read_csv(info_path, header=0)
# patients = info.name
# new_patients = [patient.split('V')[0] for patient in patients]
# cli_info['影像号'] = [str(x) for x in cli_info['影像号']]
# time = dict(zip(cli_info['影像号'], cli_info['诊断时间']))
# label = dict(zip(cli_info['影像号'], (cli_info['REICST']-1)^1))
# new_time = [time[patient] for patient in new_patients]
# new_label = [label[patient] for patient in new_patients]
#
# info['time'] = new_time
# info['label'] = new_label
# pd.DataFrame(info).to_csv(r'H:\Wei\liaoxiao\Process\Beijing_image_infoV2.csv', index=False)

# path = 'D:\CCCPatient\CCC'
# patients = os.listdir(path)
# for pat in patients:
#     subfold = os.path.join(path, pat)
#     savedata = os.path.join(subfold, 'orimeta')
#     oridata = os.path.join(subfold, 'segmeta', 'Untitled.mha')
#     shutil.copy(oridata, savedata)

# oridatapath = r'D:\ICCPatient\ICC'
# oridatapath = r'D:\CCCPatient\CCC'
# patient_path = r'G:\Radiomics\Data_process\HCC\newHCC\NEW\ICC\ICCfeature.csv'
# savepath = 'G:\Radiomics\Data_process\HCC'
# patient = pd.read_csv(patient_path, header=0)
# ID = patient.PatientID
# for pat in ID:
#     oridata = os.path.join(oridatapath, pat, 'orimeta')
#     savedata = os.path.join(savepath, 'CCC2', pat, 'orimeta')
#     shutil.copytree(oridata, savedata)

path = r'G:\Radiomics\Data_process\HCC\CCC2Output\xls\CCC2_update.csv'
features = pd.read_csv(path, header=0).columns[2:]
# index = ['glcm_maximum_cor' in fea for fea in features]
update_fea = ['glcm_maximum_cor', 'glcm_difference_variance', 'ngldm_GLN', 'ngldm_GLNN', 'ngldm_DN', 'ngldm_DNN']
used_fea = []
for fea in features:
    boo_index = [up_fea in fea for up_fea in update_fea]
    if any(boo_index) == True:
        used_fea.append(fea)

used_fea.insert(0, 'PatientID')
update_ICC = pd.read_csv(path, header=0)[used_fea]
update_ICC.to_csv(r'C:\Users\GDS\Desktop\updateCCC2feature.csv', index=False)

# if 'glcm_maximum_cor' in fea or 'glcm_difference_variance' in fea or