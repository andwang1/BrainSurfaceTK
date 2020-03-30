#!/usr/bin/env python
# coding: utf-8

# # Explore Brain Surfaces
#
# In this notebook, we load and visualise surface files
#

# In[ ]:


# image files
image_file = '/vol/biomedic2/aa16914/shared/MScAI_brain_surface/data/sub-CC00069XX12/ses-26300/anat/sub-CC00069XX12_ses-26300_T2w.nii.gz'
label_file = '/vol/biomedic2/aa16914/shared/MScAI_brain_surface/data/sub-CC00069XX12/ses-26300/anat/sub-CC00069XX12_ses-26300_drawem_all_labels.nii.gz'
# surface files
surf_file = '/vol/biomedic2/aa16914/shared/MScAI_brain_surface/data/sub-CC00069XX12/ses-26300/anat/sub-CC00069XX12_ses-26300_hemi-L_space-dHCPavg32k_pial.surf.gii'
metric_file = '/vol/biomedic2/aa16914/shared/MScAI_brain_surface/data/sub-CC00069XX12/ses-26300/anat/sub-CC00069XX12_ses-26300_hemi-L_space-dHCPavg32k_drawem.label.gii'


# In[21]:


# read image files
import SimpleITK as sitk    # SimpleITK is an image analysis package http://www.simpleitk.org/

image_sitk = sitk.ReadImage(image_file)
label_sitk = sitk.ReadImage(label_file, sitk.sitkInt8)
rescaled_image_sitk = sitk.Cast(sitk.RescaleIntensity(image_sitk), sitk.sitkUInt8)
overlay_sitk = sitk.LabelOverlay(rescaled_image_sitk, label_sitk, 0.5)
contour_sitk = sitk.LabelOverlay(rescaled_image_sitk, sitk.LabelContour(label_sitk), 1.0)


# In[ ]:

# explore and plot images

# print header information
# print(image_sitk)
# print(image_sitk.GetSize())
# print(label_sitk.GetSize())

# Extract the 3d image (numpy) array from (sitk) image file
image_array = sitk.GetArrayFromImage(image_sitk)
label_array = sitk.GetArrayFromImage(label_sitk)
overlay_array = sitk.GetArrayFromImage(overlay_sitk)
contour_array = sitk.GetArrayFromImage(contour_sitk)


# In[ ]:


# get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib
import numpy as np
import matplotlib.pyplot as plt


import nibabel as nib

img = nib.load(image_file)

a = np.array(img.dataobj)
# print(label_array.shape)
# print(label_array[100,100:200,100:200])
# raise
# In[ ]:

# NIFTs
fig = plt.figure(figsize=(10,10), facecolor='w', edgecolor='k')

fig.add_subplot(2, 2, 1)
plt.imshow(image_array[100,:,:], cmap='gray')   # show slice number 100  TOP LEFT RAW
# plt.imshow(image_array[100,100:200,100:200], cmap='gray')   # show slice number 100  TOP LEFT RAW

fig.add_subplot(2, 2, 2)
plt.imshow(label_array[100,:,:])    # show slice number 100 TOP RIGHT LABEL

fig.add_subplot(2, 2, 3)
plt.imshow(overlay_array[100,:,:])   # show slice number 100 BOTTOM LEFT OVERLAY

fig.add_subplot(2, 2, 4)
plt.imshow(contour_array[100,:,:])    # show slice number 100 BOTTOM RIGHT CONTOUR
plt.show()

# In[ ]:

# GIFTYs
from nilearn.surface import load_surf_data, load_surf_mesh

surf_mesh = load_surf_mesh(surf_file)
surf_metric = load_surf_data(metric_file)


# In[ ]:


print(surf_mesh[1].shape)
# print(surf_mesh[1])
print(np.unique(surf_metric))


# In[ ]:


from nilearn.plotting import view_surf

view = view_surf(surf_mesh, surf_metric)

view.open_in_browser() # RUN THIS FOR 3D view in browser


# In[ ]:
