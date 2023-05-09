# 3DFronts

## potential call for this method:

python  main.py --frontal_data <path/to/frontal_data_binaries> --outname <path/to/output_folder> --background_variables t q level lat lon --direction_variables u v --direction_levels 900 --background_data_fold <path/to/atmospheric_grid_data> --background_data_format %Y%m%d_%H.nc --num_sample_points 100 &

### parameter info
background_data_format is the format of the background file as a datetime conversion string. E.g. %Y%m%d_%H.nc searches for files with names such as 20160923_00.nc .

frontal_data files should be binary files containing frontal information in a lat-lon grid. Ranging from -185°E to 185°E and 90°N to -90°N at 0.25° steps. 

background_data files should be in netCDF Format

