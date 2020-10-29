import skimage
import numpy
import matplotlib.pyplot as plt
import CNN as cnn


# Load image
img = skimage.data.chelsea()
#img = skimage.data.camera()

# Converting the image into gray.
img = skimage.color.rgb2gray(img)

# 1st convolution layer
l1Filter = numpy.zeros((2,3,3))
l1Filter[0, :, :] = numpy.array([[[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]])
l1Filter[1, :, :] = numpy.array([[[1, 1, 1], [0, 0, 0], [-1, -1, -1]]])

print("\nWorking with conv layer 1")
l1FeatureMap = cnn.Conv(img, l1Filter)

print("\nReLU")
l1FeatureMapRelu = cnn.Relu(l1FeatureMap)
print("\nPooling")
l1FeatureMapReluPool = cnn.Pooling(l1FeatureMapRelu, 2, 2)
print("End of conv layer 1\n")

# 2nd convolution layer
l2Filter = numpy.random.rand(3, 5, 5, l1FeatureMapReluPool.shape[-1])

print("\nWorking with conv layer 2")
l2FeatureMap = cnn.Conv(l1FeatureMapReluPool, l2Filter)
print("\nReLU")
l2FeatureMapRelu = cnn.Relu(l2FeatureMap)
print("\nPooling")
l2FeatureMapReluPool = cnn.Pooling(l2FeatureMapRelu, 2, 2)
print("End of conv layer 2\n")

# 3rd convolution layer
l3_filter = numpy.random.rand(1, 7, 7, l2FeatureMapReluPool.shape[-1])

print("\nWorking with conv layer 3")
l3_featureMap = cnn.Conv(l2FeatureMapReluPool, l3_filter)
print("\nReLU")
l3FeatureMapRelu = cnn.Relu(l3_featureMap)
print("\nPooling")
l3FeatureMapReluPool = cnn.Pooling(l3FeatureMapRelu, 2, 2)
print("End of conv layer 3\n")

# Show results
fig0, ax0 = plt.subplots(nrows=1, ncols=1)
ax0.imshow(img).set_cmap("gray")
ax0.set_title("Input Image")
ax0.get_xaxis().set_ticks([])
ax0.get_yaxis().set_ticks([])
plt.savefig("in_img.png", bbox_inches="tight")
plt.close(fig0)

# Layer 1
fig1, ax1 = plt.subplots(nrows=3, ncols=2)
ax1[0, 0].imshow(l1FeatureMap[:, :, 0]).set_cmap("gray")
ax1[0, 0].get_xaxis().set_ticks([])
ax1[0, 0].get_yaxis().set_ticks([])
ax1[0, 0].set_title("L1-Map1")

ax1[0, 1].imshow(l1FeatureMap[:, :, 1]).set_cmap("gray")
ax1[0, 1].get_xaxis().set_ticks([])
ax1[0, 1].get_yaxis().set_ticks([])
ax1[0, 1].set_title("L1-Map2")

ax1[1, 0].imshow(l1FeatureMapRelu[:, :, 0]).set_cmap("gray")
ax1[1, 0].get_xaxis().set_ticks([])
ax1[1, 0].get_yaxis().set_ticks([])
ax1[1, 0].set_title("L1-Map1ReLU")

ax1[1, 1].imshow(l1FeatureMapRelu[:, :, 1]).set_cmap("gray")
ax1[1, 1].get_xaxis().set_ticks([])
ax1[1, 1].get_yaxis().set_ticks([])
ax1[1, 1].set_title("L1-Map2ReLU")

ax1[2, 0].imshow(l1FeatureMapReluPool[:, :, 0]).set_cmap("gray")
ax1[2, 0].get_xaxis().set_ticks([])
ax1[2, 0].get_yaxis().set_ticks([])
ax1[2, 0].set_title("L1-Map1ReLUPool")

ax1[2, 1].imshow(l1FeatureMapReluPool[:, :, 1]).set_cmap("gray")
ax1[2, 0].get_xaxis().set_ticks([])
ax1[2, 0].get_yaxis().set_ticks([])
ax1[2, 1].set_title("L1-Map2ReLUPool")

#plt.savefig("L1.png", bbox_inches="tight"); plt.close(fig1)
plt.show()

# Layer 2
fig2, ax2 = plt.subplots(nrows=3, ncols=3)
ax2[0, 0].imshow(l2FeatureMap[:, :, 0]).set_cmap("gray")
ax2[0, 0].get_xaxis().set_ticks([])
ax2[0, 0].get_yaxis().set_ticks([])
ax2[0, 0].set_title("L2-Map1")

ax2[0, 1].imshow(l2FeatureMap[:, :, 1]).set_cmap("gray")
ax2[0, 1].get_xaxis().set_ticks([])
ax2[0, 1].get_yaxis().set_ticks([])
ax2[0, 1].set_title("L2-Map2")

ax2[0, 2].imshow(l2FeatureMap[:, :, 2]).set_cmap("gray")
ax2[0, 2].get_xaxis().set_ticks([])
ax2[0, 2].get_yaxis().set_ticks([])
ax2[0, 2].set_title("L2-Map3")

ax2[1, 0].imshow(l2FeatureMapRelu[:, :, 0]).set_cmap("gray")
ax2[1, 0].get_xaxis().set_ticks([])
ax2[1, 0].get_yaxis().set_ticks([])
ax2[1, 0].set_title("L2-Map1ReLU")

ax2[1, 1].imshow(l2FeatureMapRelu[:, :, 1]).set_cmap("gray")
ax2[1, 1].get_xaxis().set_ticks([])
ax2[1, 1].get_yaxis().set_ticks([])
ax2[1, 1].set_title("L2-Map2ReLU")

ax2[1, 2].imshow(l2FeatureMapRelu[:, :, 2]).set_cmap("gray")
ax2[1, 2].get_xaxis().set_ticks([])
ax2[1, 2].get_yaxis().set_ticks([])
ax2[1, 2].set_title("L2-Map3ReLU")

ax2[2, 0].imshow(l2FeatureMapReluPool[:, :, 0]).set_cmap("gray")
ax2[2, 0].get_xaxis().set_ticks([])
ax2[2, 0].get_yaxis().set_ticks([])
ax2[2, 0].set_title("L2-Map1ReLUPool")

ax2[2, 1].imshow(l2FeatureMapReluPool[:, :, 1]).set_cmap("gray")
ax2[2, 1].get_xaxis().set_ticks([])
ax2[2, 1].get_yaxis().set_ticks([])
ax2[2, 1].set_title("L2-Map2ReLUPool")

ax2[2, 2].imshow(l2FeatureMapReluPool[:, :, 2]).set_cmap("gray")
ax2[2, 2].get_xaxis().set_ticks([])
ax2[2, 2].get_yaxis().set_ticks([])
ax2[2, 2].set_title("L2-Map3ReLUPool")


#plt.savefig("L2.png", bbox_inches="tight"); plt.close(fig2)
plt.show()

# Layer 3
fig3, ax3 = plt.subplots(nrows=1, ncols=3)
ax3[0].imshow(l3_featureMap[:, :, 0]).set_cmap("gray")
ax3[0].get_xaxis().set_ticks([])
ax3[0].get_yaxis().set_ticks([])
ax3[0].set_title("L3-Map1")

ax3[1].imshow(l3FeatureMapRelu[:, :, 0]).set_cmap("gray")
ax3[1].get_xaxis().set_ticks([])
ax3[1].get_yaxis().set_ticks([])
ax3[1].set_title("L3-Map1ReLU")

ax3[2].imshow(l3FeatureMapReluPool[:, :, 0]).set_cmap("gray")
ax3[2].get_xaxis().set_ticks([])
ax3[2].get_yaxis().set_ticks([])
ax3[2].set_title("L3-Map1ReLUPool")

#plt.savefig("L3.png", bbox_inches="tight"); plt.close(fig3)
plt.show()
