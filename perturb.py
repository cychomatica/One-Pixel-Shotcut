import numpy as np
import torch
from tqdm import tqdm


def pixel_search(clean_train_data:torch.utils.data.Dataset, pert_init=None, sparsity=1):

    H, W, C = clean_train_data.data.shape[1], clean_train_data.data.shape[2], clean_train_data.data.shape[3]
    pts = []

    if type(pert_init) in [np.ndarray]:
        perturbation = pert_init
    else:
        perturbation = np.zeros_like(clean_train_data.data, dtype=float)

    for i in range(10):

        idx_class_i = np.where(np.array(clean_train_data.targets) == i)[0]
        img_class_i = clean_train_data.data[idx_class_i] / 255

        score_class_i = np.zeros((H*W, 2**C), dtype=float)

        print('searching class {}'.format(i))
        for point in tqdm(range(len(score_class_i))):

            point_x = point // H
            point_y = point % H
            
            for pixel_value in range(2**C):

                channel_value = np.zeros(3)
                channel_value[0] = pixel_value // 2 // 2
                channel_value[1] = pixel_value // 2 % 2
                channel_value[2] = pixel_value % 2

                '''objective of searching'''
                if [point, pixel_value] in pts:
                    score_class_i[point, pixel_value] = 0
                else:
                    score_class_i[point, pixel_value] = np.mean(np.abs(channel_value - img_class_i[:,point_x,point_y,:])) \
                                                        / np.std(np.abs(channel_value - img_class_i[:,point_x,point_y,:])) 

        score_class_i_ranking = np.unravel_index(np.argsort(score_class_i, axis=None), score_class_i.shape)
        
        for i in range(sparsity):

            max_point, max_pixel_value = score_class_i_ranking[0][-i-1], score_class_i_ranking[1][-i-1]
            max_point_x, max_point_y = max_point // H, max_point %H
            max_channel_value = np.zeros(3)
            max_channel_value[0] = max_pixel_value // 2 // 2
            max_channel_value[1] = max_pixel_value // 2 % 2
            max_channel_value[2] = max_pixel_value % 2
            
            pts.append([max_point,max_pixel_value])
            perturbation[idx_class_i,max_point_x,max_point_y,:] = max_channel_value - img_class_i[:,max_point_x,max_point_y,:]

    return perturbation

