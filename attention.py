import jax.nump as jnp
def softmaxT (logits,temperature=1): #default temperature is 1
  return jnp.exp(logits/temperature) / np.sum(np.exp(logits/temperature), axis=0)
def attention(q,k,temp=1): 
  return softmaxT(jnp.dot(q,k.T),temperature=temp)
