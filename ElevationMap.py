from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pylab as pl
from PIL import Image
import numpy as np
import pylab


'''

    3D elevation visualizer of 2D thresholded image
    Based on image ROI boundaries. 
    author: Ryan Pontillas Iraola

'''

img = Image.open('kkkk.png').convert('L')
z   = np.asarray(img)

mydata = z[::1,::1]
fig = pl.figure(facecolor='w')
#subplot(nrows, ncols, plotnumber)

ax2 = fig.add_subplot(1,1,1,projection='3d')
x,y = np.mgrid[:mydata.shape[0],:mydata.shape[1]]
ax2.plot_surface(x,y,mydata,cmap=pl.cm.jet,rstride=10,cstride=10,linewidth=0,antialiased=False)
ax2.set_title('3D')
ax2.set_zlim3d(0,255)
pl.show()
