

// **ANNOTATE_AND_EXPORT**

// Open json annotation on the wsi file and then save the wsi annotated file into OME.TIFF

// To use this script you must have a .json file with exact same path as the image, but with .json extension instead
// In // setting colors sections, one has to change the name of classes with the classes athat are described in the json file
// One can set the downsampling ratio of the output (can also prevent downsampling)<<s



// to use this script you must have a .json file with exact same path as the image, but with .json extension instead

import com.google.gson.Gson
import qupath.lib.geom.Point2
import qupath.lib.objects.PathAnnotationObject
import qupath.lib.roi.PolygonROI
import qupath.lib.gui.scripting.QPEx
import java.io.Reader
import java.nio.file.Files
import java.nio.file.Paths
import java.awt.Color
import qupath.lib.common.GeneralTools;

import qupath.lib.gui.images.servers.RenderedImageServer
import java.io.File;
import qupath.lib.images.writers.ome.OMEPyramidWriter
import qupath.lib.gui.images.stores.ImageRegionStoreFactory
import qupath.lib.gui.prefs.PathPrefs
import qupath.lib.images.servers.ImageServerProvider

// adapted from https://github.com/qupath/qupath/blob/4b464e54a48765cebd710ec8b207d7f916c32913/qupath-extension-bioformats/src/main/java/qupath/lib/images/writers/ome/ConvertCommand.java
void createTileCache() {
	// TODO: Refactor this to avoid replicating logic from QuPathGUI private method
	Runtime rt = Runtime.getRuntime();
	long maxAvailable = rt.maxMemory(); // Max available memory
	if (maxAvailable == Long.MAX_VALUE) {
//		logger.warn("No inherent maximum memory set - for caching purposes, will assume 64 GB");
		maxAvailable = 64L * 1024L * 1024L * 1024L;
	}
	double percentage = PathPrefs.tileCachePercentageProperty().get();
	if (percentage < 10) {
		percentage = 10;
	} else if (percentage > 90) {
		percentage = 90;			
	}
	long tileCacheSize = Math.round(maxAvailable * (percentage / 100.0));	
	var imageRegionStore = ImageRegionStoreFactory.createImageRegionStore(tileCacheSize);
	ImageServerProvider.setCache(imageRegionStore.getCache(), BufferedImage.class);
}

// ***** plotting annotations ******

// setting colors
bg = getPathClass('Background')
bg.setColor(Color.RED.getRGB())
np = getPathClass('Neoplastic')
np.setColor(Color.RED.getRGB())
inf = getPathClass('Inflammatory')
inf.setColor(Color.GREEN.getRGB())
con = getPathClass('Connective')
con.setColor(Color.BLUE.getRGB())
dead = getPathClass('Dead')
dead.setColor(Color.YELLOW.getRGB())
nne = getPathClass('Non-Neoplastic Epithelial')
nne.setColor(Color.ORANGE.getRGB())

cls_array = [bg, np, inf, con, dead, nne]

// reading .json
img_dir = getCurrentServerPath().split('file:')[1].tokenize("[")[0]
json_dir = (GeneralTools.getNameWithoutExtension(img_dir) + ".json")
reader = Files.newBufferedReader(Paths.get(json_dir));
map = new Gson().fromJson(reader, Map)

// add annotations
annotations = []
for (annotation in map.values()) {
    vertices = annotation['contour']
    points = vertices.collect {new Point2(it[0], it[1])}
    polygon = new PolygonROI(points)
    pathAnnotation = new PathAnnotationObject(polygon)
    cls = annotation['type']
    pathAnnotation.setPathClass(cls_array[cls])
    annotations << pathAnnotation
}

QPEx.addObjects(annotations)
reader.close()

// ******** exporting flattened image to .ome.tiff *********
createTileCache()

// getting rendered image data 
def imageData = getCurrentImageData()
def server = new LabeledImageServer.Builder(imageData).build()
// server = RenderedImageServer.Builder(getCurrentImageData()).build() //para cli, creo que va por aca

// instantiate OMEPyramidWriter builder
int tileHeight=1024, tileWidth=1024  // maybe it is worth to play with this parameters
builder = new OMEPyramidWriter.Builder(server).parallelize().lossyCompression().tileSize(tileWidth, tileHeight).bigTiff(true)

// configure more stuff of the builder
// double downsample = 4
double pyramid = 2
builder.scaledDownsampling(pyramid)
builder.allZSlices();
builder.allTimePoints();
writer = builder.build();

// setting output file
output_path = (GeneralTools.getNameWithoutExtension(img_dir) + ".ome.tiff")
output_file = new File(output_path)
path = output_file.getAbsolutePath();

// write ome tiff
writer = OMEPyramidWriter.createWriter(writer)
writer.writeImage(path);

