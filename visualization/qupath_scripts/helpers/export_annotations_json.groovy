

// **EXPORT_ANNOTATION_JSON**

// Save the open annotations as a json file. As an option, it can also be saved as a
// GeoJson file. 

// the Output Path has to be set

import qupath.lib.images.servers.LabeledImageServer

def imageData = getCurrentImageData()
def name = GeneralTools.getNameWithoutExtension(imageData.getServer().getMetadata().getName())

def annotations = getAnnotationObjects()
def Outputpath = buildFilePath('/home/lsancere/These/CMMC/Ada_Mount/lsancere/Data_General/TrainingSets/Segmenter_cancer_annotation/TrainingSet184/',  name  +'.json' )


// TO create GeoJson without the 'FEATURE_COLLECTION' parameter to outputs a simple JSON object/array
// exportObjectsToGeoJson(annotations, Outputpath)

// TO create GeoJson
// 'FEATURE_COLLECTION' is standard GeoJSON format for multiple objects
exportObjectsToGeoJson(annotations, Outputpath, "FEATURE_COLLECTION")


print 'Done!'

// Try to have project path
 
//def project = getProject()
//def projectpath = project.getPath()

//public String removeLast(sun.nio.fs.UnixPath s, int n) {
//    if (null != s && !s.isEmpty()) {
//        s = s.substring(0, s.length()-n);
//    }
//    return s;
//}

// projectpath = removeLast(projectpath,8)

