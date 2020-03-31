from nilearn.plotting import view_img
import nibabel as nib


fp1 = "/home/cemlyn/Documents/Projects/MScGroupProject/data/gm_volume3d/sub-CC00055XX06_ses-9300_T2w_graymatter.nii"
fp2 = "/home/cemlyn/Documents/Projects/MScGroupProject/data/gm_volume3d/sub-CC00054XX05_ses-8800_T2w_graymatter.nii"
fp3 = "/home/cemlyn/Documents/Projects/MScGroupProject/data/gm_volume3d/sub-CC00053XX04_ses-8607_T2w_graymatter.nii"

for file_path in [fp1, fp2, fp3]:
    img = nib.load(file_path)
    img_html = view_img(img, colorbar=False, bg_img=False, cmap='gray', black_bg=False).open_in_browser()
