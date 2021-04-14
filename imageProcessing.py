import numpy as np
#expands to 56,56
def expand(image): 
  image=image.reshape(28,28)
  spaces=np.zeros((28,28))
  space=spaces.astype('float32')
  fimage=np.concatenate((image,spaces))
  fimage=np.concatenate((fimage,np.concatenate((space,space))),axis=1)
  fimage=fimage.reshape(56,56)
  return fimage

def affine_transformation(input_img,affine,sampling_grid):
  w,h=input_img.shape
  transformed_grid=jnp.matmul(affine,sampling_grid)
  transformed_grid=transformed_grid.T
  x_s = transformed_grid[:, 0:1].squeeze() #take all x
  y_s = transformed_grid[:, 1:2].squeeze() #take all y
  x = ((x_s + 1.) * w) * 0.5 #final transformed x and y coordinates which contains non-integer x and y values (1.2 etc...)
  y = ((y_s + 1.) * h) * 0.5

  #bilinear interpolation
  # grab 4 nearest corner points for each (x_i, y_i) #to grab the nearest real pixel value assign value
  x0 = np.floor(x).astype(np.int32)
  x1 = x0 + 1
  y0 = np.floor(y).astype(np.int32)
  y1 = y0 + 1
  # make sure it's inside img range [0, H] or [0, W]
  x0 = np.clip(x0, 0, w-1)
  x1 = np.clip(x1, 0, w-1)
  y0 = np.clip(y0, 0, h-1)
  y1 = np.clip(y1, 0, h-1)

  # look up pixel values at corner coords 
  Ia = input_img[y0, x0]
  Ib = input_img[y1, x0]
  Ic = input_img[y0, x1]
  Id = input_img[y1, x1]

  # calculate deltas. 
  wa = (x1-x) * (y1-y)
  wb = (x1-x) * (y-y0)
  wc = (x-x0) * (y1-y)
  wd = (x-x0) * (y-y0)
  out = wa*Ia + wb*Ib + wc*Ic + wd*Id
  return out


  #negative moves object to the left.
def animate(frame1,rep,interval=0.1): 
  w,h=frame1.shape
  
  x=np.linspace(-1,1,w)
  y=np.linspace(-1,1,h)
  xt,yt=np.meshgrid(x,y)
  one_temp= np.ones((w*h))
  samp_grid = np.vstack([xt.flatten(), yt.flatten(),one_temp])

  frame=[]
  value=0
  for i in range(rep):
    value-=interval
    affine=np.array([[1,0,value], 
                     [0,1,0]] )
    frame.append(affine_transformation(frame1,affine,samp_grid)) 
  return frame