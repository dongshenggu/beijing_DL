from __future__ import print_function

import logging
import os
import SimpleITK as sitk
import pandas as pd
# import radiomics
from radiomics import featureextractor
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import numpy as np
import shutil
from scipy.ndimage import zoom
from dicomPreprocess import findMainLesion
from collections import Counter

class ori_dicom:
    pass

class seg_dicom:
    pass

def loadDicomFile(filename):
    ds = sitk.ReadImage(filename)
    img_array = sitk.GetArrayFromImage(ds)
    sliceIndex, _, _ = np.nonzero(img_array)
    min_slice, max_slice = sliceIndex.min(), sliceIndex.max()
    range = max_slice - min_slice + 1

    return img_array.shape, range, max_slice, min_slice

def write_ori(new_image, save_path, image_size):
    # min_value, max_value = new_image.flatten().min(), new_image.flatten().max()
    min_value, max_value = 0, 1400
    for i in range(new_image.shape[0]):
        # new_imageArr = (new_image[i] - new_image[i].min()) / (new_image[i].max() - new_image[i].min())
        new_imageArr = (new_image[i] - min_value) / (max_value - min_value)
        new_imageArr[new_imageArr>1] = 1.
        new_imageArr[new_imageArr<0] = 0.
        new_imageArr = new_imageArr.astype(np.float32)
        # new_imageArr = cv2.cvtColor(new_imageArr, cv2.COLOR_GRAY2RGB)
        # new_imageArr = np.uint8(new_imageArr * 255)
        # new_imageArr = cv2.resize(new_imageArr, (image_size, image_size))
        cv2.imwrite(os.path.join(save_path, str(i) + '.png'), new_imageArr)


def write_seg(new_mask, save_path, image_size):
    for i in range(new_mask.shape[0]):
        new_imageArr = new_mask[i]
        new_imageArr = new_imageArr.astype(np.float32)
        # new_imageArr = cv2.cvtColor(new_imageArr, cv2.COLOR_GRAY2RGB)
        new_imageArr = np.uint8(new_imageArr * 255)
        # new_imageArr = cv2.resize(new_imageArr, (image_size, image_size))
        cv2.imwrite(os.path.join(save_path, str(i) + '.png'), new_imageArr)

def read_all():
    """read  the whole figure instead extracting mask region"""
    series = ['A', 'N', 'V']
    dicomPath = 'H:\Wei\liaoxiao\Process\Data'
    case_name = os.listdir(dicomPath)
    image_size = 128
    for serie in series:
        allpatient = []
        save_infopath = os.path.join('H:\Wei\liaoxiao\Process', 'full_' + serie + '.csv')
        for name in case_name:
            print(name)
            orifile = ori_dicom()
            segfile = seg_dicom()
            seriePath = os.path.join(dicomPath, name, serie)
            ori_filePath = os.path.join(seriePath, 'file_ori.nrrd')
            seg_filePath = os.path.join(seriePath, 'file_seg.nrrd')
            ds = sitk.ReadImage(ori_filePath)
            ori_img_array = sitk.GetArrayFromImage(ds)
            orifile.origin = ds.GetOrigin()  # x, y, z
            orifile.spacing = ds.GetSpacing()  # x, y, z
            ds = sitk.ReadImage(seg_filePath)
            seg_img_array = sitk.GetArrayFromImage(ds)
            sliceIndex, _, _ = np.nonzero(seg_img_array)
            segfile.min_slice, segfile.max_slice = sliceIndex.min(), sliceIndex.max()
            segfile.range = segfile.max_slice - segfile.min_slice + 1
            img = ori_img_array[segfile.min_slice:(segfile.max_slice+1)]
            mask = seg_img_array[segfile.min_slice:(segfile.max_slice+1)]
            save_oriPath = os.path.join(r'H:\Wei\liaoxiao\Process\Processed_Data_png', name, serie, 'ori')
            save_segPath = os.path.join(r'H:\Wei\liaoxiao\Process\Processed_Data_png', name, serie, 'seg')
            if not os.path.exists(save_oriPath):
                os.makedirs(save_oriPath)
            if not os.path.exists(save_segPath):
                os.makedirs(save_segPath)
            write_ori(img, save_oriPath, image_size)
            write_seg(mask, save_segPath, image_size)
            orifile.mean_pixel = img.mean()
            orifile.max_pixel = img.max()
            orifile.min_pixel = img.min()
            orifile.shape = img.shape
            orifile.patient = name
            allpatient.append([orifile.patient, orifile.shape, orifile.origin, orifile.spacing, orifile.min_pixel,
                               orifile.mean_pixel, orifile.max_pixel, segfile.min_slice, segfile.max_slice, segfile.range])
        patient = pd.DataFrame(allpatient, columns=['Patient', 'shape', 'origin', 'spacing', 'min_pixel',
                                                    'mean_pixel', 'max_pixel', 'min_slice', 'max_slice', 'range'])
        # patient.to_csv(save_infopath, index=False)

def to_sameSlcie():
    sliceNum = 15
    series = ['A', 'N', 'V']
    oridata_path = r'H:\Wei\liaoxiao\Process\Processed_Data'
    case_name = os.listdir(oridata_path)
    for serie in series:
        # save_infopath = os.path.join('H:\Wei\liaoxiao\Process', 'sameSlice_' + serie + '.csv')
        for name in case_name:
            print(name)
            ori_path = os.path.join(oridata_path, name, serie, 'ori')
            files = os.listdir(ori_path)
            newfiles = [str(i)+'.jpg' for i in list(range(len(files)))]
            volume = np.empty((len(files), 512, 512), dtype=np.uint8)
            for i, file in enumerate(newfiles):
                slice = cv2.imread(os.path.join(ori_path, file), flags=0)
                slice = slice[np.newaxis, :, :]
                volume[i] = slice  # np.append(volume, slice, axis=0)
            new_volume = zoom(volume, (sliceNum / len(files), 512 / 512, 512 / 512))
            save_oriPath = os.path.join(r'H:\Wei\liaoxiao\Process\full_sameSlice_Data', name, serie, 'ori')
            # save_segPath = os.path.join(r'H:\Wei\liaoxiao\Process\Processed_Data', name, serie, 'seg')
            if not os.path.exists(save_oriPath):
                os.makedirs(save_oriPath)
            # if not os.path.exists(save_segPath):
            #     os.makedirs(save_segPath)
            for i in range(sliceNum):
                cv2.imwrite(os.path.join(save_oriPath, str(i) + '.jpg'), new_volume[i])

def whole_to_roi(img_size, extenPix=3, seg_type='3D'):
    series = ['A', 'N', 'V']
    oridata_path = r'H:\Wei\liaoxiao\Process\Processed_Data_png'
    case_name = os.listdir(oridata_path)
    rang = []
    for serie in series:
        allpatient = []
        save_infopath = os.path.join('H:\Wei\liaoxiao\Process', 'ProcessToRoi_' + serie + '.csv')
        for name in case_name:
            save_path = os.path.join('H:\Wei\liaoxiao\Process\ProcessToRoi', name, serie, 'ori')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            print(name)
            ori_path = os.path.join(oridata_path, name, serie, 'ori')
            seg_path = os.path.join(oridata_path, name, serie, 'seg')
            files = os.listdir(ori_path)
            newfiles = [str(i)+'.png' for i in list(range(len(files)))]
            oriData = np.zeros((len(files), 512, 512), dtype=np.uint8)
            segData = np.zeros((len(files), 512, 512), dtype=np.uint8)
            for i, file in enumerate(newfiles):
                slice = cv2.imread(os.path.join(ori_path, file), flags=0)
                slice = slice[np.newaxis, :, :]
                oriData[i] = slice
                seg_slice = cv2.imread(os.path.join(seg_path, file), flags=0)
                seg_slice = seg_slice[np.newaxis, :, :]
                # print(Counter(seg_slice.flatten()))
                segData[i] = seg_slice
            imgData, min_slice, max_slice = findMainLesion(oridata=oriData, segdata=segData, extenPix=extenPix,
                                                           img_size=img_size, type=seg_type)
            # for k in range(imgData.shape[0]):
            #     cv2.imwrite(os.path.join(save_path, str(k) + '.png'), imgData[k])
            allpatient.append([name, imgData.shape[0]])
        patient = pd.DataFrame(allpatient, columns=['Patient', 'range'])
        patient.to_csv(save_infopath, index=False)


if __name__ == "__main__":
    # read_all()
    # to_sameSlcie()
    # whole_to_roi(128, extenPix=5, seg_type='3D')
    # assert 0
    series = ['A', 'N', 'V']
    dicomPath = 'H:\Wei\liaoxiao\Process\Data'
    case_name = os.listdir(dicomPath)
    allpatient = []
    image_size = 128
    for serie in series:
        save_infopath = os.path.join('H:\Wei\liaoxiao\Process', 'Normal_Data2_' + serie + '.csv')
        for name in ['1051014']:
            print(name)
            orifile = ori_dicom()
            segfile = seg_dicom()
            allfea = pd.DataFrame()
            seriePath = os.path.join(dicomPath, name, serie)
            params = r'G:\Radiomics\Py_program\beijing_DL\Beijing_CTparameter.yaml'
            extractor = featureextractor.RadiomicsFeaturesExtractor(params)
            imageName = os.path.join(seriePath, 'file_ori.nrrd')
            maskName = os.path.join(seriePath, 'file_seg.nrrd')
            if imageName is None or maskName is None:  # Something went wrong, in this case PyRadiomics will also log an error
                print('Error getting testcase!')
                exit()
            image, mask = extractor.loadImage(imageName, maskName)
            orifile.origin = image.GetOrigin()
            orifile.spacing = image.GetSpacing()
            new_image = sitk.GetArrayFromImage(image)
            # allpatient.append([name + serie, new_image.shape, new_image.min(), new_image.max()])
            new_mask = sitk.GetArrayFromImage(mask)
            save_oriPath = os.path.join(r'H:\Wei\liaoxiao\Process\Normal_Data2', name, serie, 'ori')
            save_segPath = os.path.join(r'H:\Wei\liaoxiao\Process\Normal_Data2', name, serie, 'seg')
            if not os.path.exists(save_oriPath):
                os.makedirs(save_oriPath)
            if not os.path.exists(save_segPath):
                os.makedirs(save_segPath)
            new_image = new_image[1:-1]
            new_mask = new_mask[1:-1]
            # write_ori(new_image, save_oriPath, image_size)
            # write_seg(new_mask, save_segPath, image_size)
            orifile.mean_pixel = new_image.mean()
            orifile.max_pixel = new_image.max()
            orifile.min_pixel = new_image.min()
            orifile.shape = new_image.shape
            orifile.patient = name
            sliceIndex, _, _ = np.nonzero(new_mask)
            segfile.min_slice, segfile.max_slice = sliceIndex.min(), sliceIndex.max()
            segfile.range = segfile.max_slice - segfile.min_slice + 1
            allpatient.append([orifile.patient, orifile.shape, orifile.origin, orifile.spacing, orifile.min_pixel,
                               orifile.mean_pixel, orifile.max_pixel, segfile.min_slice, segfile.max_slice, segfile.range])
        pd.DataFrame(allpatient, columns=['name', 'shape', 'origin', 'spacing', 'min_pixel',
                                          'mean_pixel', 'max_pixel', 'min_slice', 'max_slice', 'range']).to_csv(save_infopath, index=False)