def get_data_path(data_nativeness, data_compression, data_type, hemisphere='left'):
    '''
    Returns the correct path to files    data_nativeness: {'native', 'aligned'}
    data_compression: {'50', '90', '10k', '20k', '30k'}
    data_type: {'inflated', 'pial', 'midthickness', 'sphere', 'veryinflated', 'white'}
    hemisphere: {'left', 'right', 'both'}    Eg.
    data_folder = '/vol/biomedic/users/aa16914/shared/data/dhcp_neonatal_brain/surface_fsavg32k/reduced_50/vtk/pial'
    data_folder = '/vol/biomedic/users/aa16914/shared/data/dhcp_neonatal_brain/surface_native/reduced_50/inflated/vtk'
    files_ending = '_hemi-L_pial_reduce50.vtk'
    files_ending = '_left_inflated_reduce50.vtk'    For merged:
    file names : sub-CC00050XX01_ses-7201_merged_white.vtk
    path to files: /vol/biomedic/users/aa16914/shared/data/dhcp_neonatal_brain/surface_native_04152020/merged/original_native/white/vtk    # NATIVE
    left: /vol/biomedic/users/aa16914/shared/data/dhcp_neonatal_brain/surface_native_04152020/hemispheres/reducedto_30k/inflated/vtk
    merged: /vol/biomedic/users/aa16914/shared/data/dhcp_neonatal_brain/surface_native_04152020/merged/reducedto_30k/inflated/vtk
    surface_native_04152020/hemispheres/reducedto_30k/inflated/vtk/sub-CC00401XX05_ses-123900_merged_inflated.vtk    left:   sub-CC00050XX01_ses-7201_left_inflated_30k.vtk
    merged: sub-CC00050XX01_ses-7201_merged_inflated_30k.vtk
    '''
    root = '/vol/biomedic/users/aa16914/shared/data/dhcp_neonatal_brain/'
    data_nativeness_paths = {'native': 'surface_native_04152020/',
                             'aligned': 'surface_fsavg32k/'}
    hemispheres = {'both': 'merged/',
                  'left': 'hemispheres/',
                  'right': 'hemispheres/'}
    data_compression_paths = {'50': 'reduced_50/',
                              50: 'reduced_50/',
                              '90': 'reduced_90/',
                              90: 'reduced_90/',
                              'original_native': 'original_native/',
                              'original': 'original_32k/',
                              'original_aligned': 'original_32k/',
                              '10k': 'reducedto_10k/',
                              '20k': 'reducedto_20k/',
                              '30k': 'reducedto_30k/',
                              }
    data_type_paths = {'inflated': 'inflated/',
                       'pial': 'pial/',
                       'midthickness': 'midthickness/',
                       'sphere': 'sphere/',
                       'veryinflated': 'veryinflated/',
                       'white': 'white/'}
    hemisphere_paths = {'left_native_original': f'_left_{data_type}.vtk',
                        'left_native_50':       f'_left_{data_type}_reducedby50percent.vtk',
                        'left_native_90':       f'_left_{data_type}_reducedby90percent.vtk',
                        'left_native_10k':      f'_left_{data_type}_10k.vtk',
                        'left_native_20k':      f'_left_{data_type}_20k.vtk',
                        'left_native_30k':      f'_left_{data_type}_30k.vtk',
                        'right_native_original': f'_right_{data_type}.vtk',
                        'right_native_50':       f'_right_{data_type}_reducedby50percent.vtk',
                        'right_native_90':       f'_right_{data_type}_reducedby90percent.vtk',
                        'right_native_10k':      f'_right_{data_type}_10k.vtk',
                        'right_native_20k':      f'_right_{data_type}_20k.vtk',
                        'right_native_30k':      f'_right_{data_type}_30k.vtk',
                        'merged_native_original': f'_merged_{data_type}.vtk',
                        'merged_native_50':       f'_merged_{data_type}_reducedby50percent.vtk',
                        'merged_native_90':       f'_merged_{data_type}_reducedby90percent.vtk',
                        'merged_native_10k':      f'_merged_{data_type}_10k.vtk',
                        'merged_native_20k':      f'_merged_{data_type}_20k.vtk',
                        'merged_native_30k':      f'_merged_{data_type}_30k.vtk',
                        'left_aligned_original': f'_hemi-L_{data_type}.vtk',
                        'left_aligned_50':       f'_hemi-L_{data_type}_reduce50.vtk',
                        'left_aligned_90':       f'_hemi-L_{data_type}_reduce90.vtk',
                        'right_aligned_original': f'_hemi-R_{data_type}.vtk',
                        'right_aligned_50':       f'_hemi-R_{data_type}_reduce50.vtk',
                        'right_aligned_90':       f'_hemi-R_{data_type}_reduce90.vtk'}
    if hemisphere == 'both':
        _hemisphere = 'merged'
    else:
        _hemisphere = hemisphere
    if data_nativeness == 'native':
        data_folder = root + data_nativeness_paths[data_nativeness] + hemispheres[hemisphere] + data_compression_paths[data_compression] + data_type_paths[data_type] + 'vtk'

    files_ending = hemisphere_paths[f'{_hemisphere}_{data_nativeness}_{data_compression}']
    return data_folder, files_ending



