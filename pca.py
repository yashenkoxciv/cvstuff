import io
import zipfile
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

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
    with zipfile.ZipFile('pcv_data.zip') as zf:
        font_imgs_zip = io.BytesIO(
                zf.open('data/fontimages.zip').read()
        )
        #print(type(font_imgs_zip))
        with zipfile.ZipFile(font_imgs_zip) as f:
            a = []
            for t in f.namelist():
                if '.jpg' in t:
                    rawimg = f.open(t).read()
                    bimg = io.BytesIO(rawimg)
                    aimg = Image.open(bimg)
                    #print(aimg)
                    #aimg.show()
                    a.append(
                            np.array(aimg).flatten()
                    )
                    #break
    x = np.array(a)
    v, s, mx = pca(x)
    
    plt.imshow(v[0].reshape((25, 25)))
    plt.show()
    
    plt.imshow(v[1].reshape((25, 25)))
    plt.show()
    
    plt.imshow(mx.reshape((25, 25)))
    plt.show()
                
