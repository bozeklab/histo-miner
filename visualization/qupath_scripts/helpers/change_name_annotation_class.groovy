
// **CHANGE NAME ANNOTATION CLASS**

// Change the name of a given annotation class.
// To change it for all the images in the project seclect Run > Rrun for project.



// Taken from image.Sc forum, user melvingelbard
// Link here: https://forum.image.sc/t/change-name-existing-annotation-class/37955


// Written in QuPath 0.2.0-m11
def newClass = getPathClass("Tissue2")   // Your new class here
def oldClass = getPathClass("Tissue")

getAnnotationObjects().each { annotation ->
    if (annotation.getPathClass().equals(oldClass))
        annotation.setPathClass(newClass)
}
//fireHierarchyUpdate() // If you want to update the count in the Annotation pane

print "Done!"