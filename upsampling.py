import numpy as np 
import jax.numpy as jnp
import jax
from jax import jit

#channels last
def calculate_size(input_size,kernel_size=[2,2,3],stride=2): 
  ih,iw,id=input_size
  i_shape=np.array([ih,iw]) 
  kh,kw,kd=kernel_size
  kernel=np.array([kh,kw])
  return np.prod((stride*(i_shape-1)+kernel)) * kd

#You have to specify the size of the kernel even with depth that you want 
#the image to come out to
def TranposeConv_ind (input_size,kernel_size=[2,2,3],stride=2):
  ih,iw,id=input_size
  kh,kw,kd=kernel_size
  total_size=calculate_size(input_size,kernel_size,stride) 
  grid=np.array(range(total_size)) 
  s=int(np.sqrt(total_size/kd)) #this is also really bad as it assumes image is a square. 
  grid=grid.reshape(kd,s,s) 
  gd,gh,gw=grid.shape
  #draw indices
  indices=[]
  y=-stride
  x=0 
  while y<=(gh-kh-1):
    y+=stride
    x=0
    while x<=(gw-kw):
      temp=grid[:,y:y+kh,x:x+kw]
      temp=temp.flatten()
      indices.append(temp)
      x+=stride
  return np.array(indices),grid.shape

###Tranpose conv2d main function
def TranposeConv2D(input_image,param,transpose_ind,gridShape): 
  ih,iw,id=input_image.shape
  input_image=input_image.reshape(-1,id)
  calculation = jnp.dot(input_image,param.T) 
  #print('f', 'calculation shape: ', calculation.shape)
  #print('f,' 'tranpose_ind shape: ', transpose_ind.shape)
  
  #now map to upsized image. 
  zero_grid=jnp.tile(np.zeros((gridShape)),(len(transpose_ind),1,1,1))
  #print('f','zero_grid shape: ', zero_grid.shape)
  zero_grid=zero_grid.reshape(len(zero_grid),-1)

  #for i in range(len(zero_grid)): 
    #zero_grid[i,transpose_ind[i]]=calculation[i]
  zero_grid=jax.ops.index_update(zero_grid,jax.ops.index[:,transpose_ind],calculation)
    #jax.ops.index_update(test,3,6)
  output= jnp.sum(zero_grid,axis=0)
  return output
