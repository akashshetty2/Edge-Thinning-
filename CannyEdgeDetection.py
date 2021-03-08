#Do not import any additional modules
import numpy as np
from PIL.Image import open
import matplotlib.pyplot as plt

### Load, convert to grayscale, plot, and resave an image
I = np.array(open('Iribe.jpg').convert('L'))/255

# plt.imshow(I,cmap='gray')
# plt.axis('off')
# plt.show()

# plt.imsave('test.png',I,cmap='gray')

def sums(I, h, startx, starty): 
    bound = int(len(h))
    sum1 = 0 
    x = startx 
    y = starty 

    for i in range(bound): 
        y = starty 
        for j in range(bound): 
            sum1 = (I[x, y] * h[i, j]) + sum1
            y = y + 1
        x = x + 1 
    
    return sum1 

### Part 1
def gausskernel(sigma):
    #Create a 3*sigma x 3*sigma 2D Gaussian kernel


    if (sigma % 2 != 0): 
        h = np.zeros((3 * sigma, 3 * sigma))
    else: 
        h = np.zeros(((3 * sigma) + 1, (3 * sigma) + 1))

    bound = int(len(h) / 2)

    x = 0 
    for i in range(-bound, bound + 1): 
        y = 0 
        for j in range(-bound, bound + 1):  
            h[x, y] = np.exp((-((i*i) + (j*j))/(2*sigma*sigma)))/(np.pi * 2 * sigma * sigma)
            y = y + 1

        x = x + 1 

    gauss_sum = np.sum(h)
    h = np.divide(h, gauss_sum)

    # h=np.array([[1.]])
    return h


def myfilter(I,h):
    #Appropriately pad I
    #Convolve I with h

    I_filtered = np.zeros((I.shape[0], I.shape[1]))

    
    for x in range(I.shape[0]):
        for y in range(I.shape[1]):
            bound = int(len(h)/2) 
            if not(x + bound >= I.shape[0] or y + bound >= I.shape[1] or (x - bound) < 0 or (y - bound) < 0): 
                
                cornerx = x - bound
                cornery = y - bound 
                I_filtered[x,y] = sums(I, h, cornerx, cornery)



    return I_filtered


h1=np.array([[-1/9,-1/9,-1/9],[-1/9,2,-1/9],[-1/9,-1/9,-1/9]])
h2=np.array([[-1,3,-1]])
h3=np.array([[-1],[3],[-1]])


### Part 2
Sx=np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
Sy=np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
def myCanny(I,sigma=1,t_low=.5,t_high=1):
    #Smooth with gaussian kernel
    #Find img gradients
    #Thin edges
    #Hystersis thresholding
    from scipy.ndimage.measurements import label
    
    
    kernel = gausskernel(sigma)
    gauss_filter = myfilter(I, kernel) 


    image_dx = myfilter(gauss_filter, Sx)
    image_dy = myfilter(gauss_filter, Sy)

    magnitude = np.sqrt(np.add(np.square(image_dx), np.square(image_dy))) 

    theta = np.arctan2(image_dy, image_dx) 
    theta_prime = np.zeros((theta.shape[0], theta.shape[1]))
    
    for i in range(theta.shape[0]): 
        for j in range(theta.shape[1]): 
            degrees = np.degrees(theta[i][j]) 

            if (degrees >= -22.5 and degrees <= 22.5) or (degrees >= 157.5 and degrees <= 202.5): 
                theta_prime[i, j] = 0 
            elif (degrees > 22.5 and degrees <= 67.5) or (degrees > 202.5 and degrees <= 247.5) or (degrees <= -67.5 and degrees > -122.5) : 
                theta_prime[i, j] = 45 
            elif (degrees > 67.5 and degrees <= 112.5) or (degrees > 247.5 and degrees <= 292.5) or (degrees <= -122.5 and degrees > -157.5): 
                theta_prime[i, j] = 90
            elif (degrees > 112.5 and degrees <= 157.5) or (degrees > 292.5 and degrees <= 337.5): 
                theta_prime[i, j] = 135 
    
    pixel1 = 0
    pixel2 = 0 
    pixel3 = 0 

    for i in range(theta_prime.shape[0]): 
        for j in range(theta_prime.shape[1]): 
            if not(i == 0 or j == 0 or i == (theta_prime.shape[0] - 1) or j == (theta_prime.shape[1] - 1)):
                if (theta_prime[i][j] == 90): 
                    pixel1 = magnitude[(i + 1), j] 
                    pixel2 = magnitude[i, j]
                    pixel3 = magnitude[(i - 1), j]
                elif (theta_prime[i][j] == 0): 
                    pixel1 = magnitude[i, (j + 1)] 
                    pixel2 = magnitude[i, j]
                    pixel3 = magnitude[i, (j - 1)] 
                elif (theta_prime[i][j] == 135): 
                    pixel1 = magnitude[(i + 1), (j + 1)]  
                    pixel2 = magnitude[i, j]
                    pixel3 = magnitude[(i - 1), (j - 1)]
                else: 
                    pixel1 = magnitude[(i + 1), (j - 1)] 
                    pixel2 = magnitude[i, j]
                    pixel3 = magnitude[(i - 1), (j + 1)]
                
                if not(pixel2 > pixel1 and pixel2 > pixel3): 
                    gauss_filter[i, j] = 0 
                else:  
                    gauss_filter[i, j] = 1


    

    # return gauss_filter  

    for i in range(magnitude.shape[0]):
        for j in range(magnitude.shape[1]): 
            if (gauss_filter[i][j] == 1):  
                if (magnitude[i][j] < t_low): 
                    gauss_filter[i][j] = 0 
                elif (magnitude[i][j] > t_high): 
                    gauss_filter[i][j] = 1
           
    
    mag_label, components = label(gauss_filter)
    

    for comp in range(1, components + 1): 
        group = np.argwhere(mag_label == comp)
       
        cont = 0
        for coord in group: 
            if (magnitude[coord[0]][coord[1]] > t_high): 
                cont = 1
                break
        
        if (cont == 0): 
            for coord in group: 
                if (magnitude[coord[0]][coord[1]] < t_high): 
                    gauss_filter[coord[0]][coord[1]] = 0


    return gauss_filter 










edges=myCanny(I,sigma=1,t_low=.15,t_high=.3)
plt.imshow(edges, interpolation='none', cmap = 'gray')

plt.show()