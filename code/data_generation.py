import numpy as np
import os
import scipy.signal as Ssignal
from scipy.fftpack import fftshift,fft,ifft
import torch

def data_getting(path, save_path, save_name):
    data_list = os.listdir(path)
    data_npy = []
    for img_name in data_list:
        img_item_path = os.path.join(path, img_name)
        sample = np.load(img_item_path, allow_pickle=True)
        Data_Seq = sample[0, :] + 1j * sample[1, :]

        f, t, Sxx = Ssignal.spectrogram(Data_Seq, fs=2000, nperseg=32, noverlap=13,
                                        window='hamming', return_onesided=False)
        Sxx = 20 * np.log10(Sxx / np.max(Sxx))
        Zxx = np.pad(fftshift(Sxx, axes=0), pad_width=((0, 0), (1, 2)), mode='reflect')
        #         print(Zxx.shape)
        Zxx = torch.tensor(Zxx)
        Zxx = torch.nn.functional.avg_pool2d(Zxx.reshape(1, 32, 128), 3, 1, padding=1)
        Zxx = Zxx.detach().numpy().reshape(32, 128)
        Zxx[Zxx < -80] = np.min(Zxx)
        Zxx = Zxx - np.min(Zxx)
        data_npy.append(Zxx)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.save(save_path + '/' + save_name, data_npy)  # save pad data


def data_getting_train_enh(path, save_path, save_name):
    data_list = os.listdir(path)
    data_npy = []
    for img_name in data_list:
        img_item_path = os.path.join(path, img_name)
        sample = np.load(img_item_path, allow_pickle=True)
        Data_Seq = sample[0, :] + 1j * sample[1, :]

        f, t, Sxx = Ssignal.spectrogram(Data_Seq, fs=2000, nperseg=32, noverlap=13,
                                        window='hamming', return_onesided=False)
        Sxx = 20 * np.log10(Sxx / np.max(Sxx))
        Zxx = np.pad(fftshift(Sxx, axes=0), pad_width=((0, 0), (1, 2)), mode='reflect')
        Zxx = torch.tensor(Zxx)
        data_npy.append(Zxx)
        Zxx = torch.nn.functional.avg_pool2d(Zxx.reshape(1, 32, 128), 3, 1, padding=1)
        Zxx = Zxx.detach().numpy().reshape(32, 128)
        data_npy.append(Zxx)
        Zxx[Zxx < -80] = np.min(Zxx)
        Zxx = Zxx - np.min(Zxx)
        data_npy.append(Zxx)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.save(save_path + '/' + save_name, data_npy)  # save pad data


if __name__ == "__main__":
    for save_name in ['people0', 'people1', 'people2', 'people3','people4', 'people5']:
        for filename in ['pinch', 'swipe_downward', 'circle', 'swipe_from_left_to_right', 'push_in']:
            root_dir = '../dataset'
            path = os.path.join(root_dir, save_name, filename)
            save_root_dir = './data/train'
            save_path = os.path.join(save_root_dir, filename)
            data_getting_train_enh(path, save_path, save_name)

    for save_name in ['people6', 'people7']:
        for filename in ['pinch', 'swipe_downward', 'circle', 'swipe_from_left_to_right', 'push_in']:
            root_dir = '../dataset'
            path = os.path.join(root_dir, save_name, filename)
            save_root_dir = './data/test'
            save_path = os.path.join(save_root_dir, filename)
            data_getting(path, save_path, save_name)
