

# HOGDescriptor
You can also try experimenting with the OpenCV HOGDescriptor which can
be much faster than the skimage function at extracting HOG features.


# MLPClassifier
It would be good to also test some other classifiers besides SVC to
compare the performance. One other classifier that I recommend trying
is the MLPClassifier which several students have found to work well
for this project.

# Sliding Window Search
One more thing you can experiment with is thresholding the decision
function which helps to ensure high confidence predictions and reduce
false positives.

    if svc.decision_function(sample) > threshold:

The decision function returns a confidence score based on how far away
the sample is from the decision boundary. A higher decision function
means a more confident prediction so by setting a threshold on it, you
can ensure that you are only considering high confidence predictions
as vehicle detections.


