import cv2
import os
import matplotlib.pyplot as plt


class CropLayer(object):
    def __init__(self, params, blobs):
        # initialize our starting and ending (x, y) coordinate of
        # the crop
        self.startX = 0
        self.startY = 0
        self.endX = 0
        self.endY = 0

    def getMemoryShapes(self, inputs):
        (inputShape, targetShape) = (inputs[0], inputs[1])
        (batchSize, numChannels) = (inputShape[0], inputShape[1])
        (H, W) = (targetShape[2], targetShape[3])

        # compute the starting and ending crop coordinates
        self.startX = int((inputShape[3] - targetShape[3]) / 2)
        self.startY = int((inputShape[2] - targetShape[2]) / 2)
        self.endX = self.startX + W
        self.endY = self.startY + H

        return [[batchSize, numChannels, H, W]]

    def forward(self, inputs):
        return [inputs[0][:, :, self.startY:self.endY,
        self.startX:self.endX]]


if __name__ == "__main__":
    # Load HED model
    protoPath = os.path.join('hed_model', 'deploy.prototxt')
    modelPath = os.path.join('hed_model', 'hed_pretrained_bsds.caffemodel')
    net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    # Register Crop Layer
    cv2.dnn_registerLayer("Crop", CropLayer)

    # Read image
    img = cv2.imread('8027274.372160.jpg')
    (H, W) = img.shape[:2]

    # Convert image to blob data
    blob = cv2.dnn.blobFromImage(img, 
        scalefactor=1.0, 
        size=(W, H), 
        mean=(104.00698793, 116.66876762, 122.67891434), 
        swapRB=False, 
        crop=False)

    # set the blob as input to the network and perform a forward pass
    net.setInput(blob)
    hed = net.forward()
    hed = cv2.resize(hed[0, 0], (W, H))
    hed = (255*hed).astype("uint8")

    # cv2 stores image in BGR format
    # converting to RGB for pyplot
    img2 = cv2.cvtColor(hed, cv2.COLOR_BGR2RGB)

    plt.imshow(img2)
    plt.show();

    # cv2.imshow("HED", hed)
    # cv2.waitKey(0)
