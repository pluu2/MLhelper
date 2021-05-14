import jax.numpy as jnp
def softmaxT (logits,temperature=1): #default temperature is 1
  return jnp.exp(logits/temperature) / jnp.sum(jnp.exp(logits/temperature), axis=0)
def attention(q,k,temp=1): 
  return softmaxT(jnp.dot(q,k.T),temperature=temp)
#you can take the indices that the software generates.so the indices are right now set with range of 0: total_image_pixels, we need to normalize it to be between 0..1 or even
#-1,1  . Let's try 0,1
def normalize_coordinate(indices): 
  return indices/np.max(indices)
def restore_coordinate(indices,original_indices): 
   output=indices*np.max(original_indices)
   return output.astype('int32')
#Let t= position, let W_t be paramete, let p represent positional embedding. 
def generate_embeddings(indices,parameter): 
  indices=indices.reshape(len(indices),-1) #flatten the kernel into one peice
  return jnp.dot(indices,parameter.T) 

#query is some vector draw up specific values (which is a layer below). 
def attention_cluster(query,values,temperature=1,repeats =1): 
  #this needs to go into a layer. 
  #for a given head of attention. you can sum together
  #you can calculate all the heads, and all the associated cluster, multiplied by different values. 
  #ys=range(len(att)) 
  #values=layer2
  #probably can be vectorized for sure. [head,query,values]
  #query=values[head] #for a given head, hopfield like network. 
  for i in range(repeats):
    att=attention(query,values,temp=temperature) #this gives the probability distribution. 
    query=jnp.dot(att,values)
    
  #hadamand product to get the new transformation. 
  cluster=att.reshape(-1,1) * values #gather components. 
  return cluster