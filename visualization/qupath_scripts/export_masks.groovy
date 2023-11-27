

// **EXPORT_MASKS**

// The following script export a mask from a choosen annotation. The mask can be downsampled. 


// One must set the output path here "def pathOutput = buildFilePath" 

import qupath.lib.images.servers.LabeledImageServer

def imageData = getCurrentImageData()

// Define output path 
def name = GeneralTools.getNameWithoutExtension(imageData.getServer().getMetadata().getName())

//  -- > If output path is ABSOLUTE
// def pathOutput = buildFilePath('/path/to/define/', name)
def pathOutput = buildFilePath(
'/enter/path/here/', 
name)

//  -- > If Output path is linked to Project dir (RELATIVE)
// def pathOutput = buildFilePath(PROJECT_BASE_DIR, 'ExportMasks', name) 

mkdirs(pathOutput)

// Define output resolution
// double requestedPixelSize = 1.0

// Convert to downsample
// double downsample = requestedPixelSize / imageData.getServer().getPixelCalibration().getAveragedPixelSize()
double downsample = 32.0 //To follow Whole-SlideIPP Pipeline


def labelServer = new LabeledImageServer.Builder(imageData)
  .backgroundLabel(0, ColorTools.BLACK) // Specify background label (usually 0 or 255)
  .downsample(downsample)    // Choose server resolution; this should match the resolution at which tiles are exported
  .useUniqueLabels()         // Assign labels based on instances, not classifications
  .multichannelOutput(false) // If true, each label refers to the channel of a multichannel binary image (required for multiclass probability)
  .useFilter({p -> p.getPathClass() == getPathClass('tumor-region')}) 
  .build()


// Export each 

def outputPath = buildFilePath(pathOutput, 'TumorRegion_' + name + '.png')
    writeImage(labelServer, outputPath)



print 'Done!'