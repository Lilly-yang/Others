# -*- coding: utf-8 -*-

#reload_ext signature
#matplotlib inline

import cv2
import numpy as np
import os
import pydicom
import matplotlib.pyplot as plt
from glob import glob
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.ndimage
from skimage import morphology
from skimage import measure
from skimage.transform import resize
from sklearn.cluster import KMeans
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.tools import FigureFactory as FF
from plotly.graph_objs import *
init_notebook_mode(connected=True)
from pydicom.filereader import InvalidDicomError
import pandas as pd
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import skimage, os
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing
from skimage.measure import label,regionprops, perimeter
from skimage.morphology import binary_dilation, binary_opening
from skimage.filters import roberts, sobel
from skimage import measure, feature
from skimage.segmentation import clear_border
from skimage import data
from scipy import ndimage as ndi
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.misc


os.system('cls')

data_path = r"D:\Lilly\Data\CT\whole_body_data_2\du_jie\CT\\"
output_path = working_path = r"D:\Lilly\Data\CT\whole_body_data_2\du_jie"

g = glob(data_path + '/*.dcm')

g.sort(key = len)
# Print out the first 5 file names to verify we're in the right folder.
print ("Total of %d DICOM images.\nFirst 5 filenames:" % len(g))
print ('\n'.join(g[:10]))


#      
# Loop over the image files and store everything into a list.
# 

def load_scan(path):
    
    slices = []
    for s in os.listdir(path):
        try:
            one_slices = pydicom.dcmread(path + os.sep +s, force = True)
            #one_slices = dicom.read_file(data_path + os.sep +s)
        except IOError:
            print('No such file')
            continue
        except InvalidDicomError:
            print('Invalid Dicom file')
            continue
        slices.append(one_slices)
    
#    slices = [pydicom.read_file(path + '/' + s) for s in path]
#    slices = [pydicom.read_file(s) for s in g]
    slices.sort(key = lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices

def get_pixels_hu(scans):
    image = np.stack([s.pixel_array for s in scans])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 1
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    for scans_number in range(len(scans)):
        intercept = scans[scans_number].RescaleIntercept
        slope = scans[scans_number].RescaleSlope
        
        if slope != 1:
            image[scans_number] = slope * image[scans_number].astype(np.float64)
            image[scans_number] = image[scans_number].astype(np.int16)
        
        image[scans_number] += np.int16(intercept)
        
    return np.array(image, dtype=np.int16)

id=0
patient = load_scan(data_path)
imgs = get_pixels_hu(patient)

np.save(output_path + "fullimages_%d.npy" % (id), imgs)


file_used = output_path+"fullimages_%d.npy" % id
imgs_to_process = np.load(file_used).astype(np.float64) 

plt.hist(imgs_to_process.flatten(), bins=50, color='c')
plt.xlabel("Hounsfield Units (HU)")
plt.ylabel("Frequency")
plt.show()


id = 0
imgs_to_process = np.load(output_path+'fullimages_{}.npy'.format(id))

def sample_stack(stack, rows=6, cols=6, start_with=1, show_every=1):
    fig,ax = plt.subplots(rows,cols,figsize=[12,12])
    for i in range(rows*cols):
        ind = start_with + i*show_every
        if ind < len(stack):
            ax[int(i / rows), int(i % rows)].set_title('slice %d' % ind)
            ax[int(i / rows), int(i % rows)].imshow(stack[ind], cmap='gray')
            ax[int(i / rows), int(i % rows)].axis('off')

    plt.show()

sample_stack(imgs_to_process)

print ("Slice Thickness: %f" % patient[0].SliceThickness)
print ("Pixel Spacing (row, col): (%f, %f) " % (patient[0].PixelSpacing[0], patient[0].PixelSpacing[1]))



id = 0
imgs_to_process = np.load(output_path+'fullimages_{}.npy'.format(id))

def resample(image, scan, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    spacing = map(float, ([scan[0].SliceThickness] + [scan[0].PixelSpacing[0]] + [scan[0].PixelSpacing[1]]))
    spacing = np.array(list(spacing))

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)
    
    return image, new_spacing

print ("Shape before resampling\t", imgs_to_process.shape)
imgs_after_resamp, spacing = resample(imgs_to_process, patient, [1,1,1])
print ("Shape after resampling\t", imgs_after_resamp.shape)


def make_mesh(image, threshold=-300, step_size=1):

    print ("Transposing surface")
    p = image.transpose(2,1,0)
    
    print ("Calculating surface")
    verts, faces, norm, val = measure.marching_cubes(p, threshold, step_size=step_size, allow_degenerate=True) 
    return verts, faces

def plotly_3d(verts, faces):
    x,y,z = zip(*verts) 
    
    print ("Drawing")
    
    # Make the colormap single color since the axes are positional not intensity. 
#    colormap=['rgb(255,105,180)','rgb(255,255,51)','rgb(0,191,255)']
    colormap=['rgb(236, 236, 212)','rgb(236, 236, 212)']
    
    fig = FF.create_trisurf(x=x,
                        y=y, 
                        z=z, 
                        plot_edges=False,
                        colormap=colormap,
                        simplices=faces,
                        backgroundcolor='rgb(64, 64, 64)',
                        title="Interactive Visualization")
    iplot(fig)

def plt_3d(verts, faces):
    print ("Drawing")
    x,y,z = zip(*verts) 
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], linewidths=0.05, alpha=1)
    face_color = [1, 1, 0.9]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, max(x))
    ax.set_ylim(0, max(y))
    ax.set_zlim(0, max(z))
    ax.set_axis_bgcolor((0.7, 0.7, 0.7))
    plt.show()

print('imgs_after_resamp dtype', imgs_after_resamp.dtype)    
v, f = make_mesh(imgs_after_resamp, 350)
plt_3d(v, f)
    
#Standardize the pixel values
def make_lungmask(img, display=False, optimize=True):
    row_size= img.shape[0]
    col_size = img.shape[1]
    
    mean = np.mean(img)
    std = np.std(img)
    img = img-mean
    img = img/std
    # Find the average pixel value near the lungs
    # to renormalize washed out images
    middle = img[int(col_size/5):int(col_size/5*4),int(row_size/5):int(row_size/5*4)] 
    mean = np.mean(middle)  
    max = np.max(img)
    min = np.min(img)
    # To improve threshold finding, I'm moving the 
    # underflow and overflow on the pixel spectrum
    img[img==max]=mean
    img[img==min]=mean
    #
    # Using Kmeans to separate foreground (soft tissue / bone) and background (lung/air)
    #
    kmeans = KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)

    if (optimize):
        if (display):
            fig, ax = plt.subplots(2, 3, figsize=[12, 12])
            ax[0, 0].set_title("Original")
            ax[0, 0].imshow(img, cmap='gray')
            ax[0, 0].axis('off')
            
        orig_binary = img < threshold # 二值化
        orig_binary_erosion = binary_erosion(orig_binary, np.ones([3,3]))   # 腐蚀
        orig_binary_dilation = binary_dilation(orig_binary_erosion, np.ones([8,8]))  # 膨胀
        cleared = clear_border(orig_binary_dilation)        # 去除边界相连的干扰
        label_cleared = label(cleared)                        # 打标签
        # 选出两个最大的区域的标签
        areas = [r.area for r in regionprops(label_cleared)]
        areas.sort()
        if len(areas) > 2:
            for region in regionprops(label_cleared):
                if region.area < areas[-2]:
                    for coordinates in region.coords:
                           label_cleared[coordinates[0], coordinates[1]] = 0
        label_cleared = label_cleared > 0
    
        edges = roberts(label_cleared) # 获得边界
        
        edges = binary_dilation(edges, np.ones([5,5]))  #加粗
        
        binary = ndi.binary_fill_holes(edges) #内部添加
    
        get_high_vals = binary == 0
        
        img[get_high_vals] = 0
        
        if (display):
            ax[0, 1].set_title("Binary image")
            ax[0, 1].imshow(orig_binary, cmap='gray')
            ax[0, 1].axis('off')
            ax[0, 2].set_title("After Erosion and Dilation")
            ax[0, 2].imshow(orig_binary_dilation, cmap='gray')
            ax[0, 2].axis('off')
            ax[1, 0].set_title("After clear border")
            ax[1, 0].imshow(cleared, cmap='gray')
            ax[1, 0].axis('off')
            ax[1, 1].set_title("Chose 2 largest areas and label the areas")
            ax[1, 1].imshow(label_cleared)
            ax[1, 1].axis('off')
            ax[1, 2].set_title("Apply Mask on Original")
            ax[1, 2].imshow(img, cmap='gray')
            ax[1, 2].axis('off')
            plt.show()
        
    else: 
        thresh_img = np.where(img<threshold,1.0,0.0)  # threshold the image
        # First erode away the finer elements, then dilate to include some of the pixels surrounding the lung.  
        # We don't want to accidentally clip the lung.
    
        
        eroded = morphology.erosion(thresh_img,np.ones([3,3]))
        dilation_ = morphology.dilation(eroded,np.ones([8,8]))
        
        
        labels = measure.label(dilation_) # Different labels are displayed in different colors
        label_vals = np.unique(labels)
        regions = measure.regionprops(labels)
        good_labels = []
        for prop in regions:
            B = prop.bbox
            if B[2]-B[0]<row_size/10*9 and B[3]-B[1]<col_size/10*9 and B[0]>row_size/5 and B[2]<col_size/5*4:
                good_labels.append(prop.label)
        mask = np.ndarray([row_size,col_size],dtype=np.int8)
        mask[:] = 0
    
        #
        #  After just the lungs are left, we do another large dilation
        #  in order to fill in and out the lung mask 
        #
        for N in good_labels:
            mask = mask + np.where(labels==N,1,0)
        mask = morphology.dilation(mask,np.ones([10,10])) # one last dilation
        
        img = mask*img
    
        if (display):
            fig, ax = plt.subplots(2, 3, figsize=[12, 12])
            ax[0, 0].set_title("Original")
            ax[0, 0].imshow(img, cmap='gray')
            ax[0, 0].axis('off')
            ax[0, 1].set_title("Threshold")
            ax[0, 1].imshow(thresh_img, cmap='gray')
            ax[0, 1].axis('off')
            ax[0, 2].set_title("After Erosion and Dilation")
            ax[0, 2].imshow(dilation, cmap='gray')
            ax[0, 2].axis('off')
            ax[1, 0].set_title("Color Labels")
            ax[1, 0].imshow(labels)
            ax[1, 0].axis('off')
            ax[1, 1].set_title("Final Mask")
            ax[1, 1].imshow(mask, cmap='gray')
            ax[1, 1].axis('off')
            ax[1, 2].set_title("Apply Mask on Original")
            ax[1, 2].imshow(img, cmap='gray')
            ax[1, 2].axis('off')
            plt.show()
        
        
    return img


img = imgs_after_resamp[100]
make_lungmask(img, display=True ,optimize=True)

masked_lung = []

for img in imgs_after_resamp:
    masked_lung.append(make_lungmask(img))

sample_stack(masked_lung, show_every = 6)
