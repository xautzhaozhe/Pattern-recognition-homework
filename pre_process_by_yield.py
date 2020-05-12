import h5py
import os
import numpy as np
from scipy.io import loadmat
from sklearn.decomposition import PCA
import scipy.io as sio


def data_add_zero( data, patch_size=5):
        assert data.ndim == 3
        dx = patch_size // 2     # dx=2
        data_add_zeros = np.zeros(
            (data.shape[0]+2*dx, data.shape[1]+2*dx, data.shape[2]))
        data_add_zeros[dx:-dx, dx:-dx, :] = data
        return data_add_zeros


def get_patch_data(data, patch_size=5, debug=False, is_rotate=False):

        assert isinstance(data.flatten()[0], float)
        dx = patch_size // 2
        # add zeros for mirror data
        data_add_zeros = data_add_zero(data=data, patch_size=patch_size)
        # get mirror date to calculate boundary pixel，以边缘像素为轴，获取镜像
        for i in range(dx):
            data_add_zeros[:, i, :] = data_add_zeros[:, 2 * dx - i, :]
            data_add_zeros[i, :, :] = data_add_zeros[2 * dx - i, :, :]

            data_add_zeros[:, -i - 1,:] = data_add_zeros[:, -(2 * dx - i) - 1, :]
            data_add_zeros[-i - 1, :,:] = data_add_zeros[-(2 * dx - i) - 1, :, :]

        if debug is True:
            print(data_add_zeros)

        for x in range(data.shape[0]):
            for y in range(data.shape[1]):
                x_start, x_end = x, x+patch_size
                y_start, y_end = y, y+patch_size
                patch = np.array(data_add_zeros[x_start:x_end, y_start:y_end, :])

                if is_rotate:
                    # tmp: channels x patches x patches
                    tmp = patch.swapaxes(1, 2).swapaxes(0, 1)
                    rot_tmp = np.asarray(
                        [np.rot90(tmp, k=4, axes=(1, 2)) for i in range(4)])
                    # for i in range(4):
                    #     print(rot_tmp[i][0, :, :])
                    yield x, y, rot_tmp
                else:  # yield相当于一个迭代器，swapaxes函数将三维轴调换;转换后成为 2,0,1
                    yield x, y, patch.swapaxes(1, 2).swapaxes(0, 1)

def pca(data,n_band,):
    # 使用PCA把数据降维到n_band维，要是不够则填0
    row, col ,band = data.shape
    n_band = n_band
    if band==n_band:
        return data
    elif band > n_band:
        data = data.reshape(row*col,band)
        pca = PCA(n_components=n_band)
        data = pca.fit_transform(data)
        data = data.reshape(row,col,n_band)
        return data
    elif band < n_band:
        data_add_channel = np.zeros((row,col,n_band))
        data_add_channel[:,:,0:band-1] = data[:,:,0:band-1]
        return data_add_channel

def set_and_save_5d_data(dataset, patch_size = 5, is_rotate=False, n_band=188, names='PCAnames'):

    data = dataset['indian_pines_corrected']
    labels = loadmat('./hyperspectral_datas/Indian_pines_gt.mat')['indian_pines_gt']
    print('orginal data shape is: ', data.shape)
    print('label shape is: ', labels.shape)
    data, labels = np.array(data), np.array(labels)
    data = pca(data,n_band)
    print('after PCA data shape is:',data.shape)
    data_scale_to1 = data / np.max(data)        # 将数据全部归一化到 0-1
    sio.savemat(os.path.join('./hyperspectral_datas','PCA_{}.mat'.format(names)), {'data': data_scale_to1})

    data_5d = get_patch_data(data_scale_to1, patch_size=patch_size, debug=False, is_rotate=False)
    [h, w, n_channels] = data_scale_to1.shape
    n_samples = h*w*4 if is_rotate else h*w
    if is_rotate:
        h5file_name = './hyperspectral_datas/{}_patch_{}.h5'.format(names,patch_size)
    else:
        h5file_name = './hyperspectral_datas/{}_patch_{}.h5'.format(names,patch_size)

    file = h5py.File(h5file_name, 'w')
    file.create_dataset('data', shape=(n_samples, n_channels, patch_size, patch_size, 1),
                        chunks=(1024, n_channels, patch_size, patch_size, 1), dtype=np.float32,
                        maxshape=(None, n_channels, patch_size, patch_size, 1))

    file.create_dataset('labels', data=labels)
    file.close()

    with h5py.File(h5file_name, 'a') as h5f:
        for i, (x, y, patch) in enumerate(data_5d):
            if is_rotate:
                h5f['data'][4*i:4*(i+1)] = patch[:, :, :, :, None]
            else:
                h5f['data'][i] = patch[None, :, :, :, None]
        print('h5f_data shape: ', h5f['data'].shape)


if __name__ == '__main__':

    names = 'indian_pines_corrected'
    dataset = loadmat('./hyperspectral_datas/{}.mat'.format(names))
    set_and_save_5d_data(dataset, patch_size=5, is_rotate=False, n_band=224, names=names)
    print("It's all right")








