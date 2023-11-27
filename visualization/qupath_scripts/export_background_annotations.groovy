

// **EXPORT_BACKGROUND_ANNOTATIONS**

// Export the selecyed ares as png file keeping only the background and not the annotations.

import qupath.lib.images.servers.LabeledImageServer

def imageData = getCurrentImageData()

// Define output path 
def name = GeneralTools.getNameWithoutExtension(imageData.getServer().getMetadata().getName())

//  -- > If output path is ABSOLUTE
// def pathOutput = buildFilePath('/home/lsancere/These/CMMC/Local_DATA/SCC/ProcessedData/ExportSingleAnnotations/', name)

//  -- > If Output path is linked to Project dir (RELATIVE)
def pathOutput = buildFilePath(PROJECT_BASE_DIR, 'ExportRawBackground', name) 

mkdirs(pathOutput)

// Define output resolution
// double requestedPixelSize = 1.0

// Convert to downsample
// double downsample = requestedPixelSize / imageData.getServer().getPixelCalibration().getAveragedPixelSize()
double downsample = 1 //To follow Whole-SlideIPP Pipeline


// Create an ImageServer where the pixels are derived from annotations
def labelServer = new LabeledImageServer.Builder(imageData)
    .backgroundLabel(0, ColorTools.BLACK) // Specify background label (usually 0 or 255)
    .downsample(downsample)    // Choose server resolution; this should match the resolution at which tiles are exported
    .addLabel('Granulocyte', 1)      // Choose output labels (the order matters!)
    .addLabel('Lymphocyte', 2)
    .addLabel('Plasma', 3)
    .addLabel('Stroma', 4)
    .addLabel('Tumor', 5)
    .addLabel('Epithelial', 6)   
    //.lineThickness(2)          // Optionally export annotation boundaries with another label
    //.setBoundaryLabel('Boundary*', 255, ColorTools.WHITE) // Define annotation boundary label
    .multichannelOutput(false) // If true, each label refers to the channel of a multichannel binary image (required for multiclass probability)
    .build()

def server = getCurrentServer()

// Export each region
int i = 0
for (annotation in getAnnotationObjects()) {
    def region = RegionRequest.createInstance(
        labelServer.getPath(), downsample, annotation.getROI())
    i++
    def outputPath = buildFilePath(pathOutput, name + '_Region' + i + '.png')
    writeImageRegion(server, region, outputPath)
}


print 'Done!'