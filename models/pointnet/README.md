## Run instructions

### Steps for both regression and segmentation

#### 1. Prepare your data
To run Pointnet models, you mist first prepare your data. We only require that the data
1. has the *file-naming* convention specified below;
2. has the *local feature naming* conventions specified below;
3. is accompanied by a *meta file* that includes a list of all patient and session ids;
4. is accompanied by a *train/val/test data split* pickle file;
5. is of *.vtp / .vtk* file format.

**First**, the files must all be named in the following way,

```
'sub-{patient_id}_ses-{session_id}_{files_ending}'
```

* `'patient_id'` is the patient id of the subject
* `'session_id'` is the session id of the subject
* `'files_ending'` is any preferred file ending with `'.vtk'` or `'.vtp'` at the end (eg. `'_white_lefthemi_30k.vtp'`)


**Second**, your .vtk/.vtp likely contains local (per-node) features you'd like to use. To do that, make sure these arrays are named as follows:
* Brain regions labels for segmentation is `'segmentation'`
* Corrected thickness is `'corrected_thickness'`
* Curvature is `'curvature'`
* Sulcal depth is `'sulcal_depth'`
* Myelin map is `'myelin_mapl'`

These are the only local features supported at this point. Please refer to the `models/pointnet/src/dataloader.py` to add your own features. All the functions are easiliy customisable, so you will not have any problem adding your own features.

**Third**, place a file called `meta_data.tsv` in the `models/pointnet/src` folder. This tab-seperated file will be used to read all the patient and session ids during the pre-processing of the data, as well as meta data like sex or scan age.
The file should contain columns *participant_id* and *session_id*, which will be concatenated to form a unique identifier of a patient's scan.
Eg. a *meta_data.tsv* file might look like this:

```
participant_id	session_id	    gender	birth_age	birth_weight	singleton	scan_age	     scan_number	radiology_score	     sedation
CC00549XX22	    157600	    Female	42.0	        3.685	        Single	        42.142857142857	      1	                  2	                0
CC00407BN11	    124100	    Male	35.1	        2.41	        Multiple	35.5714285714285      1	                  3	                0

```

**Lastly**, make sure to include a data split file, which must be a pickle file placed in `models/pointnet/src/names.pk`. The pickle must contain a dictionary in the following format, 

```
{
'Train': ['CC00549XX22_157600', '{patient_id}_{session_id}', ... ],
'Val':   ['CC00590XX14_187000', '{patient_id}_{session_id}', ... ],
'Test':  ['CC00720XX11_211101', '{patient_id}_{session_id}', ... ]
}
```

And you're done! These are the only data-related conventions you must follow to use our PointNet toolkit. 

#### 2. Run Segmentation

Now that you have prepared your data, it's time to run the scripts. Before doing so, please follow the following steps:
1. Define the local features you want to use in the list `local_features = ['corr_thickness', ...]`. Please follow this convention: `'segmentation'` = `'drawem'`, `'corrected_thickness'` = `'corr_thickness'`, `'sulcal_depth'` = `'sulc'`, and `'myelin_mapl'` = `'myelin_map'`. For example, if you would like to include all the specified features, your `'local_features'` list should be defined as `'['drawem', 'corr_thickness', 'myelin_map', 'curvature', 'sulc']'`. 

2. Note, for segmentation, global features are not yet implemented – we hope to release it in the near future.

3. Next, specify the following variables:
* `recording` = if True, will record the experiment data.
* `REPROCESS` = if True, will reprocess the data.

4. Specify the path to the data splits

5. Specify `data_folder` – path to the folder with all the data files and `files_ending` – the endings of the files as specified in Section 1.

6. Bravo! You're ready to run PointNet++ Segmentation.

If you've set up the virtual environment from main-level README, you can run your experiment by calling the following script from the main level of the repository:

```
python3 -m scripts.segmentation.PointNet.run_pointnet_segmentation
```



### Regression
