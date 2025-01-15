

// UNDER CONSTRUCTION

// **TILE_EXTRACTOR_ANNOTATIONS_ONLY**

// Generate tiles of the annotations (white or black background to choose) in .png format 

// To use this script you must have a WSI file open with annotations
// One must set the output path here "def pathOutput = buildFilePath" and add the corresponding labels names on the "add.label" lines
// One can define the downsampling of the tiles with "double downsample =" and also the size of the tiles with ".tileSize() " 

print'Was tileSize changed accordingly?'
print'Are the extracted labels correct?'
// Just message to the user not to forget to change tileSize (to delete in general use)

import qupath.lib.images.servers.LabeledImageServer

def imageData = getCurrentImageData()
def server = getCurrentImageData().getServer()

// Define output path 
def name = GeneralTools.getNameWithoutExtension(imageData.getServer().getMetadata().getName())

//  -- > If output path is ABSOLUTE
def pathOutput = buildFilePath("//Users/lsancere/Desktop/LabelExctracted", name)


//  -- > If Output path is linked to Project dir (RELATIVE)
//def pathOutput = buildFilePath(PROJECT_BASE_DIR, 'LabelExctracted', name) 
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
    .addLabel('Granulocyte', 1)
    .addLabel('Lymphocyte', 2)
    .addLabel('Plasma', 3)
    .addLabel('Stroma', 4)
    .addLabel('Tumor', 5)
    .addLabel('Epithelial', 6) //none neoplastic
    .multichannelOutput(false)  // If true, each label is a different channel (required for multiclass probability)
    .build()

// Create an exporter that requests corresponding tiles from the original & labeled image servers
new TileExporter(imageData)
    .downsample(downsample)     // Define export resolution
    .imageExtension('.txt')     // Hugly method so far, create a txt file from raw image for it not to be generated
    .tileSize(1000, 1000)   // Define size of each tile, in pixels 
 //   .tileSize(server.Width, server.Height)   //When generating only one annotation image  
    .labeledServer(labelServer) // Define the labeled image server to use (i.e. the one we just built)
    .annotatedTilesOnly(true)  // If true, only export tiles if there is a (labeled) annotation present
    .overlap(0)                // Define overlap, in pixel units at the export resolution
    .writeTiles(pathOutput)     // Write tiles to the specified directory

print 'Done!'
