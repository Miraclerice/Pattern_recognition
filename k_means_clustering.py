# -*- coding: utf-8 -*-
# @Author: MiracleRice
# Blog   : miraclerice.com
import os.path
import time
import numpy as np
from sklearn.cluster import k_means
import matplotlib.pyplot as plt
import cv2


def get_euclidean_dis(X, Y):
    """
    calculate euclidean distance matrix of X and Y
    $$\left\Vert X_i-Y_i \right\Vert^2 = \left\Vert X_i \right\Vert^2+\left\Vert Y_i \right\Vert^2-2{X_i}\cdot{Y_i}$$
    """
    x = np.matmul(X, X.T)
    y = np.matmul(Y, Y.T)
    x_pow = np.reshape(np.diag(x), (-1, 1))
    y_pow = np.reshape(np.diag(y), (-1, 1))
    return x_pow + y_pow.T - 2 * np.matmul(X, Y.T)


def custom_kmeans(feat, n_clus, max_iter=100):
    # random initialize the feature center
    feat_cnt = np.array(feat[np.random.choice(len(feat), n_clus), :])
    label, dis = None, None
    for i in range(max_iter):
        dis = []
        # Calculate the distance from all feature points to the feature center
        for start in range(0, len(feat), 1000):
            dis.append(get_euclidean_dis(feat[start:start + 1000, :], feat_cnt))
        dis = np.concatenate(dis, axis=0)
        # get segmentation class for all feature
        label = np.argmin(dis, axis=1)
        new_cnt = []
        for j in range(n_clus):
            new_cnt.append(np.mean(feat[label == j, :], axis=0))
        new_cnt = np.array(new_cnt)
        if np.mean(abs(feat_cnt - new_cnt)) < 0.1:
            break
        # update feature center
        feat_cnt = new_cnt
    assert label is not None, 'kmeans failed'
    assert dis is not None, 'kmeans failed'
    return feat_cnt, label, dis


def color_lst():
    return [
        (0, 0, 255), (0, 255, 0), (255, 0, 0),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 128, 255), (128, 255, 128), (255, 128, 128),
        (128, 0, 255), (128, 255, 0), (255, 128, 0),
        (0, 128, 255), (0, 255, 128), (255, 0, 128),
        (64, 64, 255), (64, 255, 64), (255, 64, 64),
        (64, 0, 255), (64, 255, 0), (255, 64, 0)
    ]


def main(img_path, resize=False, feat_dim=3):
    assert feat_dim in [3, 5], 'the dimension of feature must be 3 or 5'
    # init
    n_clusters = [2, 4, 6, 8, 12, 16, 20]
    # choose ROI
    img = cv2.imread(img_path)[600:2200, 100:1700] if resize else cv2.imread(img_path)
    feat = []
    # get features b g r (x y)
    for h in range(img.shape[0]):
        for w in range(img.shape[1]):
            # normalization[0, 1]
            if feat_dim == 3:
                # b g r
                feat.append(img[h, w, :] / 255)
            else:
                # b g r x y
                feat.append(np.concatenate((img[h, w] / 255, np.array([h / img.shape[0], w / img.shape[1]])), axis=0))
    feat = np.array(feat)

    plt.figure(dpi=300)
    plt.subplot(4, 4, 1)
    plt.axis('off')
    # # Adjust title position with pad
    plt.title('Original img', pad=5)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    for i, n_clus in enumerate(n_clusters):
        start = time.time()
        # scikit-learn
        centroid, label, _ = k_means(feat, n_clusters=n_clus, n_init="auto", random_state=0)
        # opencv
        # _, label, centroid = cv2.kmeans(feat, n_clus, None,
        #                                 (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0), 3,
        #                                 cv2.KMEANS_PP_CENTERS)
        # custom
        # centroid, label, _ = k_means(feat, n_clusters=n_clus)
        print(f'n_class={n_clus}, time={time.time() - start}s')

        # Color the picture
        cor_lst = color_lst()
        label = label.reshape((img.shape[0], img.shape[1]))
        res = np.zeros(img.shape, dtype=np.uint8)
        for h in range(img.shape[0]):
            for w in range(img.shape[1]):
                res[h, w, :] = cor_lst[label[h, w]]

        plt.subplot(4, 4, i + 2)
        plt.title(f'n_class={n_clus}', pad=5)
        plt.axis('off')
        plt.imshow(res)

    plt.suptitle(f"K-means Clustering {os.path.basename(img_path).split('.')[0]}({feat_dim})")
    # Adjust the space between subplots
    plt.subplots_adjust(hspace=0.5, wspace=0.3)
    if not os.path.exists('out_img'):
        os.mkdir('out_img')
    plt.savefig(f"out_img/{os.path.basename(img_path).split('.')[0]}_{feat_dim}_{time.time()}.svg", bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main('ori_img/roommate.jpg', resize=True)
    main('ori_img/BingxianXie.jpg')
    main('ori_img/lena.png')
    main('ori_img/roommate.jpg', resize=True, feat_dim=5)
    main('ori_img/BingxianXie.jpg', feat_dim=5)
    main('ori_img/lena.png', feat_dim=5)

