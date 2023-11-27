

// **CONVERT_ANNOTATIONS_TO_DETECTIONS**

// The following tranform annotation objects to detection (never parent object in Qupath object hierarchy) objects. 

// The script is coming from this forum discussion: https://forum.image.sc/t/convert-detections-to-annotations-in-qupath/50627
// And inversing direction

def annotations = getAnnotationObjects()
def newDetections = annotations.collect {
    return PathObjects.createDetectionObject(it.getROI(), it.getPathClass())
}
removeObjects(annotations, true)
addObjects(newDetections)

print 'Done!'

