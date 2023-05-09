from Util.util import filterChannels, getCorrespondingName
import os
from Evaluation.CrossSectionApplication.CrossSectionUtils.CalculateCrossSections import getValAlongNormalWithDirs
from era5dataset.EraExtractors import MultiFileEraExtractor
import numpy as np

class CrossSectionExtractor():

    @staticmethod
    def add_processor_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("3D Cross Section Creator")
        parser.add_argument('--background_data_fold', help='path to folder containing background data. Defaults to data_fold of Data Set ')
        parser.add_argument('--background_data_format', type = str, default = None, help='format of name of background data samples. Defaults to data_format of Data Set')
        parser.add_argument('--direction_data_fold', help='path to folder containing direction data. Defaults to background_data_fold')
        parser.add_argument('--direction_data_format', type = str, default = None, help='format of name of direction data samples. defaults to background_data_format')

        parser.add_argument("--background_variables", nargs="+", type = str, default = None, help = "(atmospheric) variables from the dataset that should be used as background")
        parser.add_argument("--direction_variables", nargs=2, type = str, default = None, help = "(atmospheric) variables from the dataset that should be used for direction estimation")
        parser.add_argument("--background_levels", nargs = "+", type = int, default = None, help = "(atmospheric) levels from the dataset that should be used for background")
        parser.add_argument("--background_levelrange", nargs = "+", type = int, default = None, help = "(atmospheric) levels from the dataset that should be used for background as a range(from,to,step). This is only used if background levels None is specified")
        parser.add_argument("--direction_levels", nargs = "+", type = int, default = None, help = "(atmospheric) levels from the dataset that should be used for direction estimation")
        parser.add_argument("--direction_levelrange", nargs = "+", type = int, default = None, help = "(atmospheric) levels from the dataset that should be used for direction estimation as a range(from,to,step). This is only used if direction levels None is specified")
        parser.add_argument("--front_origin", type = str, default = "ML", help="Which type of fronts to create Cross Sections from. (ML, WS, ...)")

        parser.add_argument('--filter_low_fronts', action='store_true', help='Filter short or low valued fronts before saving')
        parser.add_argument('--target_class_labels', nargs="+", type = str, default = None, help='Reorder of model class labels to create filtered output')
        parser.add_argument('--random_points', action='store_true', help='use random points and directions instead of normals of fronts (e.g. for random sampling etc.)')
        parser.add_argument('--num_sample_points', type = int, default = 40, help='number sample Points in each direction (total number of points is then 2*distance+1)')
        parser.add_argument('--sample_point_distance', type = float, default = 20, help='distance in km between two sample points')
        
        return parent_parser

    def load_from_current_state(dataset, args):
        params = args
        # Get These values from the corresponding dataset and models instead
        params.data_fold = dataset.data_fold
        params.data_format = dataset.dataFormat
        params.out_fold = args.outname
        dict_args = vars(params)
        return CrossSectionExtractor(**dict_args)
        

    def __init__(self, out_fold, data_fold, data_format, background_data_fold = None, direction_data_fold = None, background_data_format = None, direction_data_format = None,
                    direction_variables = ["u","v"], background_variables = None, background_levels = None, background_levelrange = None, direction_levels = None,  direction_levelrange = None, front_origin = "ML", 
                    filter_low_fronts = False, class_labels = ["warm","cold","occ","stnry"], target_class_labels = None, num_sample_points = 100, sample_point_distance = 20, random_points = False, **kwargs):
        # background_data by default is located in the same folder as data
        self.background_data_fold = data_fold if background_data_fold is None else background_data_fold
        # direction_data by default is located in the same folder as background_data
        self.direction_data_fold = self.background_data_fold if direction_data_fold is None else direction_data_fold
        # How the input file name is formatted
        self.data_format = data_format
        # either set individual format for background data(if different file is used)
        # or default use the data_format
        self.background_data_format = self.data_format if background_data_format is None else background_data_format
        # either set individual format for direction data (if different file is used)
        # or default use the background_data_format (which might just be data_format if not specified)
        self.direction_data_format = self.background_data_format if direction_data_format is None else direction_data_format
        self.direction_variables = direction_variables
        self.background_variables = background_variables
        self.background_levels = background_levels
        if(self.background_levels is None and not background_levelrange is None):
            self.background_levels = [x for x in range(*background_levelrange)]
        elif((not self.background_levels is None) and (not background_levelrange is None)):
            print("background_levelrange is specified even though background_levels is specified. background_levelrange will be ignored!")
            
        self.direction_levels = direction_levels
        if(self.direction_levels is None and not direction_levelrange is None):
            self.direction_levels = [x for x in range(*direction_levelrange)]
        elif((not self.direction_levels is None) and (not direction_levelrange is None)):
            print("direction_levelrange is specified even though direction_levels is specified. background_levelrange will be ignored!")
        self.setupReader()
        self.front_origin = front_origin
        self.num_sample_points = num_sample_points
        self.sample_point_distance = sample_point_distance
        if(target_class_labels is None):
            self.target_class_labels = class_labels
        else:
            self.target_class_labels = target_class_labels
        self.model_class_labels = class_labels
        if(self.model_class_labels == self.target_class_labels):
            self.need_channel_filter = False
        else:
            self.need_channel_filter = True
        self.filter_low_fronts = filter_low_fronts
        
        self.random_points = random_points
        self.rng = np.random.default_rng(42)

        self.save_as_binary = False
        self.processInPipeline = False
        
        self.out_fold = os.path.join(out_fold, "Cross_Sections") 
        if(not os.path.exists(self.out_fold)):
            os.mkdir(self.out_fold)
        elif(os.path.isfile(self.out_fold)):
            print("Path exists and is not a directory! {}".format(self.out_fold))
            exit(1)
        for l in self.target_class_labels:
            tgt_fold = os.path.join(self.out_fold, l)
            if(not os.path.exists(tgt_fold)):
                os.mkdir(tgt_fold)
            elif(os.path.isfile(tgt_fold)):
                print("Path exists and is not a directory! {}".format(tgt_fold))
                exit(1)
        
    def __call__(self, outputs, batch, batch_idx, dataloader_idx):
        in_data,label,info,masks = batch
        
        # coordinates need to be added to the data as the last two variables
        lat_range = info[0]["lat_range"]
        lon_range = info[0]["lon_range"]
        filename = info[0]["filename"]
        resolution = info[0]["resolution"]
        
        if(self.need_channel_filter):
            outputs = filterChannels(outputs, self.model_class_labels, self.target_class_labels)
        bg = self.getBackgroundData(filename, lat_range, lon_range)
        u_dir,v_dir = self.getDirectionData(filename, lat_range, lon_range)
        fronts = self.getCorrectFrontLines(outputs)
        input_file_basename ,input_file_extension = os.path.splitext(os.path.basename(filename))
        
        
        # assert that the extracted region is oriented correctly.
        assert(lat_range[0] > lat_range[1])
        assert(lon_range[0] < lon_range[1])
        # relative origin
        baseDeg = (90, -180)

        latitude_offset_in_pix = (lat_range[0]-baseDeg[0])/resolution[0]
        longitude_offset_in_pix = (lon_range[0]-baseDeg[1])/resolution[1]
        border = 0 

        # if random points are desired 
        if(self.random_points):
            randomPoints = self.rng.integers(self.num_sample_points, [[fronts.shape[0]-self.num_sample_points],[fronts.shape[1]-self.num_sample_points]], (2,100))
            fronts *= 0
            fronts[randomPoints[0], randomPoints[1],1] = 1
            fronts = fronts[...,:2]

        crossSections = dict()
        chnlnames=["warm","cold", "occ","stnry"]
        for chnl in range(1,fronts.shape[-1]):
            channelImage = fronts[...,chnl]
            crossSections[chnlnames[chnl-1]] = []
            for label in range(1,np.max(channelImage).astype(np.int32)+1):
                filteredImage = 1*(channelImage==label)
                
                crossSection, windToNormal, normalDir, numPoints = getValAlongNormalWithDirs(filteredImage, bg, u_dir, v_dir, self.num_sample_points, self.sample_point_distance, border, (latitude_offset_in_pix,longitude_offset_in_pix), baseDeg, resolution, self.random_points, self.rng)
                savename = os.path.join(self.out_fold, self.target_class_labels[chnl-1], input_file_basename+"_{}".format(label))
                # if data is to be processed as part of a pipeline directly stream data to the next processor instead of writing to disk
                if(self.processInPipeline):
                    crossSections[chnlnames[chnl-1]].append(crossSection.astype(np.float32))
                # else save files to disk
                else:
                    if(self.save_as_binary):
                        crossSection.astype(np.float32).tofile(savename+".bin")
                        normalDir.astype(np.float32).tofile(savename+"_normalDir.bin")
                    else:
                        np.save(savename+".npy", crossSection.astype(np.float32), allow_pickle=False)
                        np.save(savename+"_normalDir.npy", normalDir.astype(np.float32), allow_pickle=False)

        if(self.processInPipeline):
            return crossSections, os.path.join(self.out_fold, input_file_basename+".nc")
        else:
            return [] 

    def getBackgroundData(self, input_filename, lat_range, lon_range):
        background_file = getCorrespondingName(input_filename, self.data_format, self.background_data_format, 0)
        background_file = os.path.join(self.background_data_fold, background_file)
        
        if(not os.path.exists(background_file)):
            print("background data: {} does not exist".format(background_file))
            return None, None, None, None, None, None

        # Generally no gradient (finite differences should be calculated)
        self.reader.multiLevelFileIdentifier = self.background_data_format
        var = self.getData(background_file, self.background_variables, lat_range, lon_range, self.background_levels)
        
        return var

    def getDirectionData(self, input_filename, lat_range, lon_range):
        direction_file = getCorrespondingName(input_filename, self.data_format, self.direction_data_format, 0)
        direction_file = os.path.join(self.direction_data_fold, direction_file)
        if(not os.path.exists(direction_file)):
            print("direction Data: {} does not exist".format(direction_file))
            return None, None, None, None, None, None
        
        

        udir, vdir = self.readDirections(direction_file, self.direction_variables, lat_range, lon_range, self.direction_levels)
        return udir, vdir

    def setupReader(self):
        # reads plain netCDF data, without any normalization
        self.reader = MultiFileEraExtractor(normType = None, ftype=1)
        
    
    def getData(self, filename, variables, lat_range, lon_range, levels = None):
        self.reader.setVariables(variables)
        self.reader.setLevels(levels)
        return self.reader(filename, lat_range, lon_range)

    def readDirections(self, filename, variables, lat_range, lon_range, levels = None):
        self.reader.multiLevelFileIdentifier = self.direction_data_format
        data = self.getData(filename, variables, lat_range, lon_range, levels)
        if(levels is None):
            levelVals = self.reader.getDims(filename)[-1]
            if(levelVals is None):
                levelrange = 1
            else:
                levelrange = len(levelVals)
        else:
            levelrange = len(levels)
        udir, vdir = data[:levelrange].mean(axis=0), data[levelrange:].mean(axis=0)
        return udir, vdir


    def getCorrectFrontLines(self, frontImage):
        # create individual fronts:
        from scipy.ndimage import binary_dilation, label
        labeled_thin = np.zeros_like(frontImage)
        for chnl in range(frontImage.shape[-1]):
            channelImg = frontImage[...,chnl]
            wide_img = binary_dilation(channelImg, iterations = 2)
            
            labeled_wide, num_features = label(wide_img)
            labeled_wide = labeled_wide*channelImg
            labeled_wide[:self.num_sample_points] = 0
            labeled_wide[:,:self.num_sample_points] = 0
            labeled_wide[-self.num_sample_points:] = 0
            labeled_wide[:,-self.num_sample_points:] = 0
            idx = 1
            thresh = 5
            for i in range(1,num_features+1):
                if((labeled_wide==i).sum() > thresh):
                    labeled_thin[labeled_wide==i,chnl] = idx
                    idx+=1

        return labeled_thin

    def finalize(self):
        pass
        
