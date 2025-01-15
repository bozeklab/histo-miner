

// **CONVERT_DETECTIONS_TO_ANNOTATIONS**

// The following tranform detection objects (never parent object in Qupath object hierarchy) to annotation objects. 

// The script is coming from this forum discussion: https://forum.image.sc/t/convert-detections-to-annotations-in-qupath/50627

def detections = getDetectionObjects()
def newAnnotations = detections.collect {
    return PathObjects.createAnnotationObject(it.getROI(), it.getPathClass())
}
removeObjects(detections, true)
addObjects(newAnnotations)

print'Done'
