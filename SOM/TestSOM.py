from SOM import SOM
import numpy as np
import matplotlib.pyplot as plt

 
#Training inputs for RGBcolors
colors = np.array([[0., 0., 0.],[0., 0., 1.],[0., 0., 0.5],[0.125, 0.529, 1.0],[0.33, 0.4, 0.67],[0.6, 0.5, 1.0],[0., 1., 0.],[1., 0., 0.],[0., 1., 1.],[1., 0., 1.],[1., 1., 0.],[1., 1., 1.],[.33, .33, .33],[.5, .5, .5],[.66, .66, .66]])
colorNames = ['black', 'blue', 'darkblue', 'skyblue','greyblue', 'lilac', 'green', 'red','cyan', 'violet', 'yellow', 'white','darkgrey', 'mediumgrey', 'lightgrey']
 
#Train a 20x30 SOM with 400 iterations
som = SOM(m=20, n=30, dim=3, epochs=400)
som.train(colors)
 
#Get output grid
imageGrid = som.get_centroids()
 
#Map colours to their closest neurons
mapped = som.map_vects(colors)
 
#Plot
plt.imshow(imageGrid)
plt.title('Color SOM')
for i, m in enumerate(mapped):
    plt.text(m[1], m[0], colorNames[i], ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5, lw=0))
plt.show()

