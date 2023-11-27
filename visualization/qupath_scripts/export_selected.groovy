

// EXPORT_SELECTED_OBJECT


// Write the region of the image corresponding to the currently-selected object



import qupath.lib.images.servers.LabeledImageServer

def imageData = getCurrentImageData()


// Define output path 
def name = GeneralTools.getNameWithoutExtension(imageData.getServer().getMetadata().getName())

//  -- > If output path is ABSOLUTE
def pathOutput = buildFilePath('/enter/path/here/region.tif')

//  -- > If Output path is linked to Project dir (RELATIVE)
// def pathOutput = buildFilePath(PROJECT_BASE_DIR, 'ExportSelected', name + '.tif') 

mkdirs(pathOutput)
print(pathOutput)


def server = getCurrentServer()

def roi = getSelectedROI()
def requestROI = RegionRequest.createInstance(server.getPath(), 1, roi)
writeImageRegion(server, requestROI, pathOutput)

print 'Done!'