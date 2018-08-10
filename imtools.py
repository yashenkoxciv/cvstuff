import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def histeq(im, nbr_bins=256):
    imhist, bins = np.histogram(im.flatten(), nbr_bins, normed=True)
    cdf = imhist.cumsum()
    cdf = 255*cdf / cdf[-1]
    im2 = np.interp(im.flatten(), bins[:-1], cdf)
    return im2.reshape(im.shape), cdf

def compute_average(imlist):
    averageim = np.array(Image.open(imlist[0]).convert('L'), 'f')
    for imname in imlist[1:]:
        try:
            averageim += np.array(Image.open(imname))
        except:
            print(imname, 'skipped')
    averageim /= len(imlist)
    return np.array(averageim, 'uint8')

def pca(x):
    # x matrix with training data stored as flattened arrays in rows
    num_data, dim = x.shape
    
    # center of data
    mean_x = x.mean(axis=0) # mean on columns
    x = x - mean_x # same shape as in x
    
    if dim > num_data:
        m = np.dot(x, x.T) # covariance matrix
        e, ev = np.linalg.eigh(m) # eigenvalues and eigenvectors
        tmp = np.dot(x.T, ev).T # this is the compact trick
        v = tmp[::-1] # reverse since last eigenvectors are the ones we want
        s = np.sqrt(e)[::-1] # reverse since eigenvalues are in increasing order
        for i in range(v.shape[1]):
            v[:, i] /= s
    else:
        u, s, v = np.linalg.svd(x)
        v = v[:num_data] # only makes sense to return the first num_data
    # projection matrix, the variance and the mean
    return v, s, mean_x

if __name__ == '__main__':
    # test histogram equalization
    img = Image.open('imgs/1_mountain.jpg').convert('L')
    #img.show()
    img2 = Image.fromarray(histeq(np.array(img))[0])
    #img2.show()
    # test averaging
    img = Image.fromarray(compute_average(
            [
                    'imgs/2_building.jpg',
                    'imgs/1_mountain.jpg',
                    'imgs/3_flower.jpg'
            ]
    ))
    #img.show()
    xs = np.random.randn(10, 784)
    pca(xs)
    

