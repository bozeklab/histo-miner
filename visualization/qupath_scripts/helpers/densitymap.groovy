

// **DENSITYMAPS**

// Generate a density maps with specific parameters

// ONGOING CODING

/

import qupath.lib.images.servers.LabeledImageServer

def imageData = getCurrentImageData()
def server = getCurrentImageData().getServer()

// Define output path 
def name = GeneralTools.getNameWithoutExtension(imageData.getServer().getMetadata().getName())



def labelServer = new LabeledImageServer.Builder(imageData)
    .backgroundLabel(0, ColorTools.BLACK) // Specify background label (usually 0 or 255)
//     .downsample(downsample)    // Choose server resolution; this should match the resolution at which tiles are exported
//    .addLabel('Background', 1)      // Choose output labels (the order matters!)
//    .addLabel('Neoplastic',2)
//    .addLabel('Connective', 3)     // "Stroma" here means non-tumor
//    .addLabel('Dead',4)
//    .addLabel('Inflammatory',5)
    .addLabel('Granulocyte', 1) 
    .addLabel('Lymphocyte', 2) 
    .addLabel('Plasme', 3) 
    .addLabel('Tumor', 4) 
    .addLabel('Stroma', 5) 
    
    .multichannelOutput(false)  // If true, each label is a different channel (required for multiclass probability)
    .build()


// Creata list of parameters for the density map to be created with these specific ones


def densityserver = new DensityMaps.DensityMapParameters(imageData)
    .getDensityType(Object %)
    .getMainObjectFilter('Tumor')
    .getMaxHeight()   
    .getMaxWidth()
    .getRadius(50)


writeDensityMapImage(imageData, densityserver, '/home/lsancere/Bureau/temp/test.png')
