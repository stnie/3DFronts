import os



class PipelineEvaluator:
    def __init__(self, parts):
        self.parts = parts
        for part in self.parts:
            part.processInPipeline = True
        self.skip_existing = False
    
    def __call__(self, *args, **kwargs):
        filename  = args[1][2][0]["filename"]
        print("handling file {}".format(filename))
        input_file_basename ,input_file_extension = os.path.splitext(os.path.basename(filename))
        if self.skip_existing and os.path.exists(os.path.join(self.parts[0].out_fold, input_file_basename+".nc")):
            print("skipping file. Already exists!")
            return None
        else:
            imd = self.parts[0](*args)
            if not imd[0] is None:
                for part in self.parts[1:]:
                    imd = part(*imd)
            return imd

    def finalize(self):
        pass


import numpy as np
class EvaluationPipeline:
    def __init__(self, parts, args):
        self.parts = parts
        for part in self.parts:
            part.processInPipeline = True
        self.allResults = dict()
        self.split_location = args.splitLocation
        self.outname = ""
        self.useSplitData = args.splitType == "algorithm"
        self.outFold = os.path.join(args.outname, "MyResults")

    
    def __call__(self, *args, **kwargs):
        filename = args[1][2][0]["filename"]
        print("handling file {}".format(filename))
        input_file_basename ,input_file_extension = os.path.splitext(os.path.basename(filename))
        if os.path.exists(os.path.join(self.parts[0].out_fold, input_file_basename+".skip")):
            print("skipping file. Already exists!")
            return None
        else:
            data, outname = self.parts[0](*args)
            if not data is None:
                # first file gives info about name
                if(self.outname == ""):
                    self.outname = outname
                for i in data:
                    if i in ["warm","cold"]:
                        if len(data[i])>0:
                            tmp_dat = dict()
                            tmp_dat[i] = data[i].copy()
                            result = self.parts[1](tmp_dat, self.useSplitData, filename, i, self.split_location)
                            self.collectResult(result,i)
            return None

    def collectResult(self, result, ev_type):
        allResults = result
        if not ev_type in self.allResults:
            self.allResults[ev_type] = np.zeros((*allResults.shape[:-1],0))
        if allResults.shape[0] > 0:
            self.allResults[ev_type] = np.concatenate((self.allResults[ev_type], allResults), axis= -1)
    def finalize(self):
        import os
        for t in self.allResults:
            basename = os.path.basename(self.outname)
            fn, ext = os.path.splitext(basename)
            if self.useSplitData:
                np.save(os.path.join(self.outFold,"results{}_{}.npy".format(fn,t)), self.allResults[t], allow_pickle = False)
            else:
                np.save(os.path.join(self.outFold,"Random_results{}_{}.npy".format(fn,t)), self.allResults[t], allow_pickle = False)
