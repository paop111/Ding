# Ding
import numpy as np
import matplotlib.pyplot as plt

numlterations = 50
V = np.matrix([[5,3,2,1],
               [4,0,8,1],
               [1,1,3,5],
               [1,6,0,4],
               [0,1,5,0]])
[n, m]= np.shape(V)
r= 2
np.random.seed(0)
W_init = np.abs(np.random.randn(n, r))
H_init = np.abs(np.random.randn(r, m))

def New_H(W,V,H):
    a1=np.matrix(W).T*V
    b1=np.matrix(W).T*np.matrix(W)*np.matrix(H)
    c1=np.array(H)*np.array(a1)/np.array(b1)
    return c1

def New_W(W,V,H):
    a2=V*H.T
    b2=np.matrix(W)*np.matrix(H)*np.matrix(H).T
    c2=np.array(W)*np.array(a2)/np.array(b2)
    return c2

def loss_function(W,V,H):
    a3=V-np.matrix(W)*np.matrix(H)
    loss=np.sum(np.array(a3)*np.array(a3))
    return loss

loss=[]
H=New_H(W_init,V,H_init)
W=New_W(W_init,V,H)
loss.append(loss_function(W,V,H))
for i in range(49):
            H=New_H(W,V,H)
            W=New_W(W,V,H)
            loss.append(loss_function(W,V,H))
            


plt.plot(np.arange(1,51,1),loss,'bx-')
plt.xlabel('Number of iteration')
plt.ylabel('Distortion')
plt.title('Loss function')
plt.savefig('Loss function of NMF.jpg', format='jpg', dpi=1000)
plt.show()
