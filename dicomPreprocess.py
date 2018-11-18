import SimpleITK as sitk
from PIL import Image
import pydicom
import numpy as np
import cv2
import os
import shutil
import matplotlib.pyplot as plt
import time
import pandas as pd
from skimage import measure
import scipy

def list_all_files(rootdir):
    import os
    _files = []
    list = os.listdir(rootdir) #列出文件夹下所有的目录与文件
    for i in range(0, len(list)):
           path = os.path.join(rootdir,list[i])
           if os.path.isdir(path):
              _files.extend(list_all_files(path))
           if os.path.isfile(path):
              _files.append(path)
    return _files

def loadFile(filename):  # writerFileName
    # segfile = os.path.join(filename, os.listdir(filename)[0])
    ds = sitk.ReadImage(filename)
    img_array = sitk.GetArrayFromImage(ds)
    sliceIndex, _, _ = np.nonzero(img_array)
    min_slice, max_slice = sliceIndex.min(), sliceIndex.max()
    range = max_slice - min_slice + 1

    return img_array.shape, range, max_slice, min_slice

def loadFiles(path, patient):
    print(patient)
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(path)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    image_array = sitk.GetArrayFromImage(image)  # z, y, x
    origin = image.GetOrigin()  # x, y, z
    spacing = image.GetSpacing()  # x, y, z
    # plt.hist(image_array.flatten(), bins=100)
    # plt.savefig(os.path.join(r'H:\CRPC\CRPC\Hist\T2', patient + '.png'))
    # plt.close()
    mean_pixel = image_array.mean()
    max_pixel = image_array.max()
    min_pixel = image_array.min()
    # Standardization(image_array, min_pixel, max_pixel, patient)
    # new_image_array = (image_array - min_pixel) / (max_pixel - min_pixel)
    # slices = new_image_array.shape[0]
    # img = new_image_array[int(slices/2)]
    # img = cv2.resize(img, (1024, 1024))
    # im = Image.fromarray(np.uint8(img * 255))
    # save_path = r'H:\CRPC\CRPC\Process\Standardization-resize'
    # im.save(os.path.join(save_path, patient + '.png'))
    return [image_array.shape, origin, spacing, mean_pixel, max_pixel, min_pixel]

def Standardization(image_array, min_pixel, max_pixel, patient):
    new_image_array = (image_array - min_pixel)/(max_pixel - min_pixel)
    save_path = os.path.join(r'H:\CRPC\CRPC\Process\Standardization', patient, 'ADC')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i in range(new_image_array.shape[0]):
        im = Image.fromarray(np.uint8(new_image_array[i]*255))
        im.save(os.path.join(save_path, str(i) + '.png'))

def loadFileInformation(filename):
    information = {}
    ds = pydicom.read_file(filename, force=True)
    information['PatientID'] = ds.PatientID
    information['PatientName'] = ds.PatientName
    information['PatientBirthDate'] = ds.PatientBirthDate
    information['PatientSex'] = ds.PatientSex
    information['StudyID'] = ds.StudyID
    information['StudyDate'] = ds.StudyDate
    information['StudyTime'] = ds.StudyTime
    information['InstitutionName'] = ds.InstitutionName
    information['Manufacturer'] = ds.Manufacturer
    information['SpacingBetweenSlices'] = ds.SpacingBetweenSlices
    information['PixelSpacing'] = ds.PixelSpacing
    information['SliceThickness'] = ds.SliceThickness
    return information

def save_image(ds, SavePath, pixel_array):
    newImage = sitk.GetImageFromArray(pixel_array)
    newImage.CopyInformation(ds)
    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()
    writer.SetFileName(SavePath)
    writer.Execute(ds)


def writerNrrd(File, writerFileName):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(File)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    image_array = sitk.GetArrayFromImage(image)
    if image_array.min() == -3024:
        image_array[image_array != image_array.min()] += 1024
        image_array[image_array == image_array.min()] = 0
    else:
        image_array += 1024
    newImage = sitk.GetImageFromArray(image_array)
    newImage.CopyInformation(image)
    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()
    writer.SetFileName(writerFileName)  # writerFileName
    writer.Execute(newImage)
    return image_array.min(), image_array.max()

def writerNrrd2(oriFile, writerFileName):
    files = os.listdir(oriFile)
    num = len(files)
    ds = sitk.ReadImage(os.path.join(oriFile, files[0]))
    img_array = sitk.GetArrayFromImage(ds)
    _, row, col = img_array.shape
    all_array = np.zeros((num, row, col))
    for i, file in enumerate(files):
        reader = sitk.ImageFileReader()
        reader.SetFileName(os.path.join(oriFile, file))
        image = reader.Execute()
        image_array = sitk.GetArrayFromImage(image)
        all_array[i] = image_array
        # all_array = np.append(all_array, image_array, axis=0)
    img = sitk.GetImageFromArray(all_array)
    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()
    writer.SetFileName(writerFileName)
    writer.Execute(img)

def write_segnrrd(filename, writerFileName):  # writerFileName
    ds = sitk.ReadImage(filename)
    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()
    writer.SetFileName(writerFileName)
    writer.Execute(ds)

def process_seg(patient_path, patients, seg_savepath):
    allpatient = []
    for pat in patients:
        # pat = pat.replace(' ', '_')
        print(pat)
        # pat = pat + '\\' + pat
        path = os.path.join(patient_path, pat+'\\V')
        allfiles = os.listdir(path)
        index_bool = ['.mha' in file for file in allfiles]
        seg_index = np.where(np.array(index_bool) == True)[0][0]
        seg_file = os.path.join(path, allfiles[seg_index])
        shape, range, max_slice, min_slice = loadFile(seg_file)
        allpatient.append([pat, shape, range, max_slice, min_slice])
    patient = pd.DataFrame(allpatient, columns=['Patient', 'shape', 'range', 'max_slice', 'min_slice'])
    patient.to_csv(seg_savepath, index=False)

def process_ori(patient_path, patients, ori_savepath):
    allpatient = []
    for pat in patients:
        print(pat)
        # pat = pat + '\\' + pat
        path = os.path.join(patient_path, pat+'\\V')
        [shape, origin, spacing, mean_pixel, max_pixel, min_pixel] = loadFiles(path, pat)
        file = os.path.join(path, os.listdir(path)[1])
        ds = pydicom.read_file(file)
        allpatient.append([pat, ds.PatientID, ds.PatientName, ds.StudyDate, ds.PatientAge, ds.PatientSex, ds.RescaleIntercept,
                            ds.RescaleSlope, shape, origin, spacing, mean_pixel, max_pixel, min_pixel])
    patient = pd.DataFrame(allpatient, columns=['Patient', 'PatientID', 'PatientName', 'StudyDate', 'PatientAge', 'PatientSex',
                                                'RescaleIntercept', 'RescaleSlope', 'shape', 'origin', 'spacing', 'mean_pixel', 'max_pixel', 'min_pixel'])
    patient.to_csv(ori_savepath, index=False)

def combined_files():
    path1 = r'C:\Users\GDS\Desktop\patient_seginfo_P.csv'
    path2 = r'C:\Users\GDS\Desktop\patient_seginfo2_P.csv'
    seg_info1 = pd.read_csv(path1)
    seg_info2 = pd.read_csv(path2)
    all_seginfo = pd.concat((seg_info1, seg_info2), axis=0)
    all_seginfo.to_csv(r'C:\Users\GDS\Desktop\allpatient_seginfo_P.csv', index=False)

def getFileOrderByUpdate(path):
    file_list = os.listdir(path)
    path_dict = {}
    for file in file_list:
        path_dict[file] = os.path.getmtime(os.path.join(path, file))
    sort_list = sorted(path_dict.items(), key=lambda e:e[1], reverse=True)
    sortedFiles = [sort[0] for sort in sort_list]
    return sortedFiles

# find the main lesion
def mainLesion2D(oridata, segdata, extenPix, img_size):
    sliceIndex, _, _ = np.nonzero(segdata)
    min_slice, max_slice = sliceIndex.min(), sliceIndex.max()
    imgdata = np.zeros((0, img_size, img_size, 3), dtype=np.float32)
    for index in range(min_slice, max_slice+1):
        slice = segdata[index]
        labels = measure.label(slice, connectivity=2)  #8连通区域标记
        regionNum = labels.max()
        if regionNum == 0:
            continue
        regionProps = measure.regionprops(labels)
        areNums = [regionProps[i].area for i in range(regionNum)]
        min_row, min_col, max_row, max_col = regionProps[np.argmax(areNums)].bbox
        if ((min_row - extenPix) >= 0) and ((min_col - extenPix) >= 0):
            slice = oridata[index][(min_row - extenPix):(max_row + extenPix), (min_col - extenPix):(max_col + extenPix)]
            # slice = cv2.normalize(slice, dst=slice, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            slice = (slice - np.mean(slice)) / np.std(slice)
            slice = cv2.cvtColor(slice, cv2.COLOR_GRAY2RGB)
            slice = cv2.resize(slice, (img_size, img_size))
            slice = slice[np.newaxis, :, :]
            imgdata = np.append(imgdata, slice, axis=0)
        else:
            print('the ROI of segdata too close to edge')
            return
    return imgdata, min_slice, max_slice

def mainLesion3D(oridata, segdata, extenPix, img_size):
    labels = measure.label(segdata, connectivity=2)  # 8连通区域标记
    imgdata = np.zeros((0, img_size, img_size, 3), dtype=np.float32)
    regionNum = labels.max()
    if regionNum == 0:
        print('segdata has not ROI')
        return
    regionProps = measure.regionprops(labels)
    areNums = [regionProps[i].area for i in range(regionNum)]
    if not segdata.shape[0] == 1:
        min_slice, min_row, min_col, max_slice, max_row, max_col = regionProps[np.argmax(areNums)].bbox
    else:
        min_row, min_col, max_row, max_col = regionProps[np.argmax(areNums)].bbox
        min_slice = 0
        max_slice = 1
    if ((min_row - extenPix) >= 0) and ((min_col - extenPix) >= 0):
        oridata = oridata[min_slice:max_slice, min_row - extenPix:max_row + extenPix, min_col - extenPix:max_col + extenPix]
        for slice in oridata:
            # slice = cv2.normalize(slice, dst=slice, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            # slice = ((slice - np.mean(slice))/np.std(slice)).astype(np.float32)
            slice = cv2.cvtColor(slice, cv2.COLOR_GRAY2RGB)
            slice = cv2.resize(slice, (img_size, img_size))
            slice = slice[np.newaxis, :, :]
            imgdata = np.append(imgdata, slice, axis=0)
    else:
        print('the ROI of segdata too close to edge')
        return
    return imgdata, min_slice, max_slice

def findMainLesion(oridata, segdata, extenPix, img_size, type):
    if type == '2D':
        imgdata, min_slice, max_slice = mainLesion2D(oridata, segdata, extenPix, img_size)
    elif type == '3D':
        imgdata, min_slice, max_slice = mainLesion3D(oridata, segdata, extenPix, img_size)
    else:
        print('the input lesion type is wrong!')
        return
    return imgdata, min_slice, max_slice

def resample(image, scan, new_spacing=[1, 1, 1]):
    spacing = np.array(scan[2], scan[0], scan[1], dtype=np.float32)
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    return image, new_spacing

def readData(img_size, extenPix=3, seg_type='3D'):
    ori_path = r'H:\Wei\liaoxiao\Process\Processed_Data\1028851\A\ori'
    seg_path = r'H:\Wei\liaoxiao\Process\Processed_Data\1028851\A\seg'
    files = os.listdir(ori_path)
    newfiles = [str(i) + '.jpg' for i in list(range(len(files)))]
    oriData = np.zeros((len(files), 512, 512), dtype=np.uint8)
    segData = np.zeros((len(files), 512, 512), dtype=np.uint8)
    for i, file in enumerate(newfiles):
        slice = cv2.imread(os.path.join(ori_path, file), flags=0)
        slice = slice[np.newaxis, :, :]
        oriData[i] = slice  # np.append(volume, slice, axis=0)
        seg_slice = cv2.imread(os.path.join(seg_path, file), flags=0)
        seg_slice = seg_slice[np.newaxis, :, :]
        segData[i] = seg_slice
    imgData, min_slice, max_slice = findMainLesion(oridata=oriData, segdata=segData, extenPix=extenPix,
                                                   img_size=img_size, type=seg_type)
    for i in range(imgData.shape[0]):
        cv2.imwrite(os.path.join(r'C:\Users\GDS\Desktop\myRoi', str(i) + '.jpg'), imgData[i])



if __name__ == "__main__":
    '''read seriestime in metainfo'''
    # patient_path = r'H:\Wei\liaoxiao\20181031renminyiyuan\buchongtuxiang'
    # 'delete .DS_Store files'
    # all_files = list_all_files(patient_path)
    # index = ['.DS_Store' in file for file in all_files]
    # allindex = np.where(np.array(index) == True)[0].tolist()
    # [os.remove(all_files[index]) for index in allindex]
    # patients = os.listdir(patient_path)
    # ori_savepath = r'C:\Users\GDS\Desktop\buchong_patient_info_V.csv'
    # seg_savepath = r'C:\Users\GDS\Desktop\buchong_patient_seginfo_V.csv'
    # allpatient = []
    # process_ori(patient_path, patients, ori_savepath)
    # process_seg(patient_path, patients, seg_savepath)
    readData(img_size=128, extenPix=5, seg_type='3D')
    assert 0

    info = []
    series = ['A', 'N', 'V']
    for serie in series:
        # dicomPath = r'H:\Wei\liaoxiao\2018-9-27FCZ\2018-9-27FCZ'
        dicomPath = r'H:\Wei\liaoxiao\20181031renminyiyuan\buchongtuxiang'
        savedicomPath = r'H:\Wei\liaoxiao\Process\Data1'
        case_name = os.listdir(dicomPath)
        for name in case_name:  # [:40]
            oriFile = os.path.join(dicomPath, name, serie)
            file = os.path.join(oriFile, os.listdir(oriFile)[1])
            ds = pydicom.read_file(file)
            allfiles = os.listdir(os.path.join(dicomPath, name, serie))
            seg_index = ['.mha' in file for file in allfiles].index(True)
            segFile = os.path.join(dicomPath, name, serie, allfiles[seg_index])
            writerOriFile = os.path.join(savedicomPath, name, serie, 'file_ori.nrrd')
            writerSegFile = os.path.join(savedicomPath, name, serie, 'file_seg.nrrd')
            if os.path.exists(oriFile) and os.path.exists(segFile):
                print(oriFile)
                if not os.path.exists(os.path.join(savedicomPath, name, serie)):
                    os.makedirs(os.path.join(savedicomPath, name, serie))
                min_pixel, max_pixel = writerNrrd(oriFile, writerOriFile)
                write_segnrrd(segFile, writerSegFile)
                info.append([name + serie, min_pixel, max_pixel])
        pd.DataFrame(info, columns=['name', 'min_array', 'max_array']).to_csv(r'C:\Users\GDS\Desktop\buchongtuxiang.csv', index=False)
    # assert 0

    # info = []
    # series = ['arterial', 'plain', 'portal vein']
    # save_series = ['A', 'N', 'V']
    # dicomPath = r'H:\Wei\liaoxiao\solution\solution\zxy_save2'
    # savedicomPath = r'H:\Wei\liaoxiao\Process\Data1'
    # case_name = os.listdir(dicomPath)
    # for serie, save_serie in zip(series, save_series):
    #     for name in case_name:  # case_name[40:]:  #
    #         oriFile = os.path.join(dicomPath, name, 'SDY00000', serie)
    #         file = os.path.join(oriFile, os.listdir(oriFile)[1])
    #         ds = pydicom.read_file(file)
    #         allfiles = os.listdir(os.path.join(dicomPath, name, 'SDY00000', serie))
    #         seg_index = ['.mha' in file for file in allfiles].index(True)
    #         # segFile = os.path.join(dicomPath, name, 'SDY00000', serie, allfiles[allfiles.index('.mha')])
    #         segFile = os.path.join(dicomPath, name, 'SDY00000', serie, allfiles[seg_index])
    #         writerOriFile = os.path.join(savedicomPath, str(ds.PatientID), save_serie, 'file_ori.nrrd')
    #         writerSegFile = os.path.join(savedicomPath, str(ds.PatientID), save_serie, 'file_seg.nrrd')
    #         if os.path.exists(oriFile) and os.path.exists(segFile):
    #             print(oriFile)
    #             if not os.path.exists(os.path.join(savedicomPath, str(ds.PatientID), save_serie)):  # +'_'+ds.SeriesDate
    #                 os.makedirs(os.path.join(savedicomPath, str(ds.PatientID), save_serie))
    #             min_pixel, max_pixel = writerNrrd(oriFile, writerOriFile)
    #             write_segnrrd(segFile, writerSegFile)
                # info.append([name + save_serie, min_pixel, max_pixel])
    # pd.DataFrame(info, columns=['name', 'min_array', 'max_array']).to_csv(r'C:\Users\GDS\Desktop\zxy_save2_arrayinfo.csv', index=False)









