from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
import scipy.signal
import copy
import cv2

#This class is used in the getGradient function to get two return values from it i.e. magnitude and orientation of the image signal and its corrosponding pixels
class ReturnValue:
    def __init__(self,magnitude,orientation):
        self.magnitude=magnitude
        self.orientation=orientation

def refine(flag,sigma,I):
    if flag:
        return cv2.GaussianBlur(I, (3,0), sigma)
    else:
        return cv2.GaussianBlur(I, (0,3), sigma)
#This function is used to get the gradient and orientation of the image pixels
def get_gradient(Ix,Iy):
    l1=[]
    l2=[]
    for i in range(len(Ix)):
        l1.append(math.sqrt(Ix[i]**2+Iy[i]**2))
        l2.append(math.atan2(Iy[i],Ix[i]))
    return ReturnValue(l1,l2)

#Used to show the images
def show_image(image,title,sigma):
    plt.imshow(image,cmap=cm.gray) 
    plt.title(title+' (σ='+str(sigma)+')')
    plt.show()

#To get the GAUSSIAN distribution array
def Gauss(sigma,n):
    r = range(-int(n/2),int(n/2)+1)
    return [1 / (sigma * math.sqrt(2*math.pi)) * math.exp(-float(x)**2/(2*sigma**2)) for x in r]

#Returns True if Non - Maximum Supression condition is valid
def check_nms(val,x1,y1,x2,y2,ar):
    return val<=ar[x1][y1] or val<=ar[x2][y2]

#Mannual Convolution method
def con(ar,G):
    ans=[]
    temp=[]
    temp.append(ar[0])
    for i in ar:
        temp.append(i)
    temp.append(ar[len(ar)-1])
    
    for i in range(len(ar)):
        sum=0
        for j in range(len(G)):
            if i+j<len(temp):
                sum+=temp[i+j]*G[j]
            else:
                break
        ans.append(sum)
    return ans

#Depth-first-search to find the neighbouring high value pixels
def dfs(i,j,ar,visited,high_pixels,shape,H):
    if ar[i][j]>=H or ar[i][j] in high_pixels:
        return True
    if ar[i][j] in visited:
        return False
    
    visited.add(ar[i][j])

    if i+1<shape[0] and dfs(i+1,j,ar,visited,high_pixels,shape,H):
        high_pixels.add(ar[i+1][j])
        return True
    if i-1>=0 and dfs(i-1,j,ar,visited,high_pixels,shape,H):
        high_pixels.add(ar[i-1][j])
        return True
    if  j+1<shape[1] and dfs(i,j+1,ar,visited,high_pixels,shape,H):
        high_pixels.add(ar[i][j+1])
        return True
    if  j-1>=0 and dfs(i,j-1,ar,visited,high_pixels,shape,H):
        high_pixels.add(ar[i][j-1])
        return True
    if  j+1<shape[1] and i+1<shape[0] and dfs(i+1,j+1,ar,visited,high_pixels,shape,H):
        high_pixels.add(ar[i+1][j+1])
        return True
    if  j-1>=0 and i+1<shape[0] and dfs(i+1,j-1,ar,visited,high_pixels,shape,H):
        high_pixels.add(ar[i+1][j-1])
        return True
    if  j+1<shape[1] and i-1>=0 and dfs(i-1,j+1,ar,visited,high_pixels,shape,H):
        high_pixels.add(ar[i-1][j+1])
        return True
    if  j-1>=0 and i-1>=0 and dfs(i-1,j-1,ar,visited,high_pixels,shape,H):
        high_pixels.add(ar[i-1][j-1])
        return True
    return False


#CANNY EDGE DETECTOR
def canny(img,sigma):

    #Taking the image and converting it to Grayscale (in case of RGB image is fetched)
    image=Image.open(img).convert('L')
    I=np.asarray(image)
    Shape=np.shape(I)
    
    show_image(I,'Original Image',sigma)

    #Gaussian distribution
    G=Gauss(3,sigma)

    #Gaussian smoothing along X direction
    Gx=[]
    for i in range(Shape[0]):
        temp=np.correlate(I[i,:], G,'same')
        #temp=con(I[i,:], G,)
        Gx.append(temp)
    Gx=refine(True,sigma,I)
    show_image(Gx,'Gaussian filtering in X direction',sigma)

    #Gaussian smoothing along Y direction
    Gy=[]
    I_tranposed=np.transpose(I)
    for i in range(Shape[1]):
        temp=np.correlate(I_tranposed[i,:], G,'same')
        #temp=con(I_tranposed[i,:], G,)
        Gy.append(temp)
    Gy=np.transpose(Gy)
    Gy=refine(False,sigma,I)
    show_image(Gy,'Gaussian filtering in Y direction',sigma)

    #Getting the resultant filtered image
    smoothed_image=[]
    for i in range(Shape[0]):
        temp=get_gradient(np.array(Gx)[i,:],np.array(Gy)[i,:])
        smoothed_image.append(temp.magnitude)
    show_image(smoothed_image,'Gaussian filtered Image',sigma)

    #Using first order derivative
    central_diff=[-1.,0.,1.]
    I=np.array(smoothed_image)
    Ix=[]
    Iy=[]

    #Calculating the first order derivative of the smoothed image along X direction
    for i in range(Shape[0]):
        temp=np.convolve(I[i,:], central_diff,'same')
        Ix.append(temp)

    show_image(Ix,'X direction Gassuian Convolution',sigma)

    #Calculating the first order derivative of the smoothed image along Y direction
    I_transposed=np.transpose(I)
    for i in range(Shape[1]):
        temp=np.convolve(I_transposed[i,:], central_diff,'same')
        Iy.append(temp)

    Iy=np.transpose(Iy)
    show_image(Iy,'Y direction Gassuian Convolution',sigma)

    #Calculating the gradient and orientation of the image from Ix and Iy -> S=Sqrt( Ix^2 + Iy^2 )
    magnitude=[]
    orientation=[]
    for i in range(Shape[0]):
        temp=get_gradient(np.array(Ix)[i,:],np.array(Iy)[i,:])
        magnitude.append(temp.magnitude)
        orientation.append(temp.orientation)

    show_image(magnitude,'Gradient Magnitude',sigma)

    thin_edges=np.copy(magnitude)

    #Non maximum supression
    Shape=np.shape(thin_edges)
    for i in range(Shape[0]):
        for j in range(Shape[1]):
            if (0<orientation[i][j] and orientation[i][j]<=22.5) or (157.5<orientation[i][j] and orientation[i][j]<=180):
                if j-1>=0 and j+1<Shape[1] and check_nms(thin_edges[i][j],i,j-1,i,j+1,thin_edges):
                    thin_edges[i][j]=0
                elif (22.5<orientation[i][j] and orientation[i][j]<=67.5):
                    if i-1>=0 and j-1>=0 and i+1<Shape[0] and j+1<Shape[1] and check_nms(thin_edges[i][j],i-1,j-1,i+1,j+1,thin_edges):
                        thin_edges[i][j]=0
                elif (67.5<orientation[i][j] and orientation[i][j]<=112.5):
                    if i-1>=0 and i+1<Shape[0] and check_nms(thin_edges[i][j],i-1,j,i+1,j,thin_edges):
                        thin_edges[i][j]=0
                elif (112.5<orientation[i][j] and orientation[i][j]<=157.5):
                    if i+1<Shape[0] and j-1>=0 and i-1>=0 and j+1<Shape[1] and check_nms(thin_edges[i][j],i+1,j-1,i-1,j+1,thin_edges):
                        thin_edges[i][j]=0

    show_image(thin_edges,'Image after Non-max Supression',sigma)

    #Hysteresis Thresholding
    final_output=np.copy(thin_edges)

    #Declaring arbitraty threshold values
    H=60 
    L=50
    high_pixels=set()
    for i in range(Shape[0]):
        for j in range(Shape[1]):
            if final_output[i][j]<=L: #If the pixel value is lower than the lower bound threshold then it is NOT an edge
                final_output[i][j]=0
            elif final_output[i][j]>H:  #If the pixel value is higher than the upper bound threshold then it is an edge
                high_pixels.add(final_output[i][j])
                final_output[i][j]=255
            else:
                #Doing Depth first search to know if the pixel is connected with any high value pixel or not
                visited=set()
                if dfs(i,j,final_output,visited,high_pixels,Shape,H):
                    final_output[i][j]=255
                else:
                    final_output[i][j]=0
                

    show_image(final_output,'Final Image',sigma)




# Calling the Canny Edge Detector with 'Image name' and 'Sigma' value as parameters
sigma=[0.8,2,5]
image=['star','church','bird']

for i in sigma:
    canny(image[0]+'.jpeg',i)


canny(image[1]+'.jpeg',sigma[0])

canny(image[2]+'.jpeg',sigma[0])
