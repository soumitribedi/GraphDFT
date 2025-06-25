import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

methods=["vxc","vxch"]
colors = ["#EC7063","#AF7AC5","#5499C7","#52BE80","#F4D03F","#EB984E","#60151D","#090E76","#AF0505","#FF00FF","#157405"]


n = len(methods)

#  along an axis plot
for i in range(n):
    name = methods[i]
    print("Reading {} ".format(name))
    with open(name,"r") as file1:
        lines1=file1.readlines()
    Z = []
    D = []
    for j in range(len(lines1)):
        x,y,z,den = lines1[j].split()
        if abs(float(x))<0.000000001 and abs(float(y))<0.000000001:
            D.append(float(den))
            Z.append(float(z))

    Z, D = zip(*sorted(zip(Z, D)))
    #plt.xlim([0,5])
    #plt.ylim([-10,10])
    if name=="rho_wf":
        for k in range(len(Z)):
            if Z[k]>1.99 and Z[k]<4.01:
                print(Z[k],D[k])
    plt.axvline(x=0,linestyle='dashed',color='grey')
    plt.axvline(x=6.02784,linestyle='dashed',color='grey')
    plt.xlabel('r (bond axis) (Bohr)',fontsize=15)
    plt.ylabel('$v_{XC}$ (a.u.)',fontsize=15)
    plt.plot(Z,D,color=colors[i],linestyle='solid',markersize=2,marker='o',label=methods[i])
    #plt.plot(Z,D,color='darkblue',linestyle='solid',markersize=3.5,label=None)

plt.legend()
plt.show()
'''
# on a plane plot
for i in range(n):
    name = 'vxc_50_14_'+methods[i]+'_Td'
    print("Reading {} ".format(name))
    with open(name,"r") as file1:
        lines1=file1.readlines()
    X = []
    Y = []
    D = []
    for j in range(len(lines1)):
        x,y,z,den = lines1[j].split()
        if abs(float(y))<0.0001 and abs(float(x))<8.0 and abs(float(z))<8.0 :
            D.append(float(den))
            X.append(float(x))
            Y.append(float(z))

    X, Y, D = zip(*sorted(zip(X,Y,D)))
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.scatter3D(X,Y,D,c=D,cmap="viridis",linewidth=1.5)
    #ax.set_xlim(-5,5)
    #ax.set_ylim(-5,5)
    #ax.set_zlim(0,50)
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('rho')
    #ax.view_init(azim=0, elev=90)

plt.legend()
plt.show()
'''
