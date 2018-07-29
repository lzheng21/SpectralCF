import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot(x, y, n_users, n_items,filename):
    index = [0] + list(range(n_users,n_users+n_items))
    x, y = x[index], y[index]
    n_users = 1
    n=[]
    for i in range(n_users+n_items):
        if i < n_users:
            n.append('$u_'+str(i+1) +'$')
        else:
            n.append('$i_' + str(i-n_users + 1) + '$')
    c = ['r','g','g','g','g']
    fig, ax = plt.subplots()
    ax.scatter(x, y,c=c,s=400)


    for i, txt in enumerate(n):
        fc = 'red' if i==0 else 'green'

        label = plt.annotate(
            txt,
            xy=(x[i], y[i]), xytext=(-20, 20),
            textcoords='offset points',ha='right', va='bottom',
            bbox=dict(boxstyle='circle,pad=0.5', fc=fc, alpha=0.5),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    plt.axis([-1, 1, -1, 1])
    plt.savefig(filename,format='eps',dpi=1000,bbox_inches='tight')
    plt.show()
    plt.close()


num = 10
n_users, n_items = 3, 4
R = [[1,0,0,0],[1,1,0,1],[1,0,1,1]]
R = np.array(R, dtype=np.float32)

A = np.zeros([n_users+n_items, n_users+n_items], dtype=np.float32)
A[:n_users, n_users:] = R
A[n_users:, :n_users] = R.T
D_inverse = np.diag(1.0/np.sum(A,axis=0))
D_inverse_half = np.diag(1.0/np.sqrt(np.sum(A,axis=0)))

D = np.diag(np.sum(A,axis=0))
L = np.identity(n_users+n_items) - np.dot(D_inverse, A)
lamda, U = np.linalg.eig(L)
index = [i[0] for i in sorted(enumerate(lamda),key=lambda i: i[1])]
lamda = lamda[index]
print(lamda)
U = U[:,index]

plot(U[:,1],U[:,2],n_users, n_items,'fig.eps')
