

// **EXPORT_FULL_ANNOTATED_IMAGE**

// Generate annotations of a whole image (white or black background to choose) in .ome.tif format. The input image size
// Should be not so high.

// To use this script you must have a WSI file open with annotations
// One must set the output path here "def pathOutput = buildFilePath" and add the corresponding labels names on the "add.label" lines
// One can define the downsampling of the tiles with "double downsample =" and also the size of the tiles with ".tileSize() " 

print'Was tileSize changed accordingly?'
// Just message to the user not to forget to change tileSize (to delete in general use)

save_path = '/home/lsancere/Bureau/' 

import qupath.lib.images.servers.LabeledImageServer

def imageData = getCurrentImageData()

// Define output path
def name = GeneralTools.getNameWithoutExtension(imageData.getServer().getMetadata().getName())

//  -- > If output path is ABSOLUTE
def pathOutput = buildFilePath("/home/lsancere/Bureau/LabelExctracted/", name + "-labels.ome.tif")

//  -- > If Output path is linked to Project dir (RELATIVE)
// def pathOutput = buildFilePath(PROJECT_BASE_DIR, 'LabelExctracted', name + '-labels.ome.tif')

mkdirs(pathOutput)

// DOWNSAMPLE
// a) You can calculate the downsample factor based on requestedPixelSize

// Define output resolution in µm
// double requestedPixelSize = 5.0
// double downsample = requestedPixelSize / imageData.getServer().getPixelCalibration().getAveragedPixelSize()

// b) or you can specify it yourself (1.0 means original magnification)
double downsample = 1.0 // original mag = 40 ; downsampled mag = 40/8 = 5

// Create an ImageServer where the pixels are derived from annotations
def labelServer = new LabeledImageServer.Builder(imageData)
  .backgroundLabel(0, ColorTools.WHITE) // Specify background label (usually 0 or 255)
//  .downsample(downsample)    // Choose server resolution; this should match the resolution at which tiles are exported
  .addLabel('Granulocyte', 1)      // Choose output labels (the order matters!)
  .addLabel('Lymphocyte', 2)
  .addLabel('Plasma', 3)
  .addLabel('Stroma', 4)
  .addLabel('Tumor', 5)
  .addLabel('Epithelial', 6)   
  .multichannelOutput(false) // If true, each label refers to the channel of a multichannel binary image (required for multiclass probability)
  .build()

// Write the image
writeImage(labelServer, pathOutput)

print 'Done!'