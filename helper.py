from skimage.filters import frangi, sobel
from skimage import exposure
from skimage.morphology import erosion, square

from scipy import ndimage as ndi
from sklearn.decomposition import PCA

import numpy as np

equalize = lambda img : exposure.equalize_adapthist(img, clip_limit=0.03)

def featureExtractionFBP(img):
   len = 90
   img = equalize(img)
   img2=img[len:-len,len:-len]

   img = frangi(img2)
   hist, rr = np.histogram(img, bins=100)
   img = img > rr[10]
   img = erosion(img, square(3))

   label_objects, nb_labels = ndi.label(img)
   sizes = np.bincount(label_objects.ravel())
   sizes[0]=0
   mask_sizes = sizes >= 400#np.max(sizes) #& (sizes<40000)#((np.max(sizes)+np.mean(sizes))*0.3)
   filled_cleaned = mask_sizes[label_objects]

   img1 = erosion(sobel(filled_cleaned)>0,square(2))

   img3 = sobel(img2)
   img2 = img3>0.1
   img2 = erosion(img2,square(2))

   kp =[]
   for idx, ii in enumerate(img2):
      for idxy, jj in enumerate(ii):
         if jj:
            kp.extend([[idx,idxy]])
   kp=np.array(kp)
   cp = np.mean(kp,axis=0)

   feature = []
   bins = 10

   points_img1=[]

   for idxx, i in enumerate(img1):
      for idxy, val in enumerate(i):
         if val:
            points_img1.extend([[idxx,idxy]])
   points_img1 = np.array(points_img1)

   dist_p1 = np.sqrt(np.sum((points_img1-cp)**2,axis=1))
   hist1_img1, _ = np.histogram(dist_p1, bins=bins,
                               range=(0,220),
                               density=True)

   feature.extend(hist1_img1)
   feature.extend([np.var(dist_p1)])

   reduced_data = PCA(n_components=1).fit_transform(points_img1)
   hist2_img1, _ = np.histogram(reduced_data, bins=bins,
                                density=True)
   feature.extend(hist2_img1)
   feature.extend([np.mean(img1)])

   return np.array(feature)

def featureExtractionTP(img):
    img2 = equalize(img)
    len=90

    img2=img2[len:-len,len:-len]
    img3 = sobel(img2)
    img2 = img3>0.1
    img2 = erosion(img2,square(2))

    kp =[]
    for idx, ii in enumerate(img2):
        for idxy, jj in enumerate(ii):
            if jj:
                kp.extend([[idx,idxy]])
    kp=np.array(kp)
    cp = np.mean(kp,axis=0)
    dist1 = np.sqrt(np.sum((kp-cp)**2,axis=1))
    hist,_ = np.histogram(dist1,bins=22,range=(0,220), density=True)
    reduced_data = PCA(n_components=1).fit_transform(kp)
    hist2_img1, _ = np.histogram(reduced_data, bins=22,
                                 density=True)
    hist = np.append(hist,hist2_img1)
    return np.append(hist,[np.mean(img2)])
