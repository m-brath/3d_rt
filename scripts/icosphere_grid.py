# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 11:08:27 2016

@author: Manfred Brath
"""
#Functions to create an icosahedric grid and to refine it
#
#   The vertices/grid points are given by an N x 3 matrix with N the number of
#   vertices/grid point. Each grid point lies on unit sphere. The columns are
#   the cartesian x, y and z coordinate of each grid point/vertex
#   There is an additional M x 3 aray with the vertex index of each triangle of
#   icosahedric grid.




import numpy as np
#from numba import jit

#==============================================================================
#  find nearest vertex
#==============================================================================


def find_nearest_Nvertices(vertices, D, N=1):
    
    #Function to find the nearest N vertices. 
    #   vertices are an M x 3 array with the cartesian coordinates of the vertices
    #   D is a vector of the search point in cartesian coordinates
    #   N is the number of the nearest vertices
    

    distance=np.zeros(np.size(vertices,axis=0))
    for i in range(0,np.size(vertices,axis=1)):

       distance=(vertices[:,i]-D[i])**2+distance;


    distance=np.sqrt(distance)

    #get the three most similar vertices
    idx=np.argsort(distance)
    distance=distance[idx]


    #we need only the first three vertices
    idx=idx[0:N]
    distance=distance[0:N]
    

    return idx, distance



#==============================================================================
#  create icosahedric grid
#==============================================================================

def make_icosahedric_grid():

    #Function to create an icosahedric_grid


    #golden number
    Z=(1+np.sqrt(5))/2;
    N=0;
    X=1;

    # vertices of the icosaeder
    vertices=np.array([[ -X, N, Z],
              [ X, N, Z],
              [-X, N,-Z],
              [ X, N,-Z],
              [ N, Z, X],
              [ N, Z,-X],
              [ N,-Z, X],
              [ N,-Z,-X],
              [ Z, X, N],
              [-Z, X, N],
              [ Z,-X, N],
              [-Z,-X, N]])


    #rescale the icosaeder, so that the norm of each grid point is unity
    norm=np.sqrt(np.sum(vertices**2,1))

    vertices=vertices/np.conj(np.transpose(np.tile(norm, (3, 1))))


    triangles=np.array([[0,4,1],
               [0,9,4],
               [9,5,4],
               [4,5,8],
               [4,8,1],
               [8,10,1],
               [8,3,10],
               [5,3,8],
               [5,2,3],
               [2,7,3],
               [7,10,3],
               [7,6,10],
               [7,11,6],
               [11,0,6],
               [0,1,6],
               [6,1,10],
               [9,0,11],
               [9,11,2],
               [9,2,5],
               [7,2,11]])
    #%vertices for each triangle
    triangles=triangles;


    return vertices, triangles





#==============================================================================
# refine icosahedric grid
#==============================================================================


def refine_icosahedric_grid(vertices, triangles, numRefinements):

    #   Function to refine an icosahedric_grid by bisecting each line and
    #   projecting the new grid point on a unit sphere


    for k in range(1,numRefinements+1):


        #number of vertices and triangles (aka facets)
        maxnumVertices=(np.size(vertices,axis=0)-2)*4+2;
        maxnumTriangles=np.size(triangles,axis=0)*4;

        #allocate
        verticesIndex_new=np.ones([maxnumVertices], dtype=int)
        triangles_new=np.ones([maxnumTriangles,3], dtype=int)
        vertices_new=np.ones([maxnumVertices,3])*np.NaN

        #Number of vertices from the refinement
        numVertices=0;

        #get size of triangles array
        [q0,q1]=np.shape(triangles)

        # go through the list of triangles
        for i in range(0,q0):


            #the three corners of the triangle
            A=vertices[triangles[i,0],:]
            B=vertices[triangles[i,1],:]
            C=vertices[triangles[i,2],:]

            #the new (refined) vertices
            BA=(B+A)/np.sqrt((B+A)@np.transpose(B+A))
            CB=(C+B)/np.sqrt((C+B)@np.transpose(C+B))
            AC=(A+C)/np.sqrt((A+C)@np.transpose(A+C))

            vertices_temp=np.array([A,B,C,BA,CB,AC])


            #==================================================================
            # now do the housekeeping, important to avoid double point
            #==================================================================

            #indexing the points of the refined grid
            index=np.arange(0,6)+numVertices;


            # Check if the vertices already exist in the list
            flag=np.zeros((len(index)), dtype=bool)

            for j in range(0,np.size(vertices_temp,axis=0)):

                logic1=vertices_temp[j,0]==vertices_new[:,0]
                logic2=vertices_temp[j,1]==vertices_new[:,1]
                logic3=vertices_temp[j,2]==vertices_new[:,2]

                logic=logic1*logic2*logic3

                if np.sum(logic)>0:
                   fidx=np.where(logic==1)
                   index[j]=fidx[0][0]
                   flag[j]=True

            # take onlythe vertices, which are not in the list.
            vertices_temp=vertices_temp[np.logical_not(flag),:]

            # the number of new vertices
            numVertices_temp=np.size(vertices_temp,axis=0)

            #replace the double ones with the already existing ones
            index[np.logical_not(flag)]=numVertices+np.arange(0,numVertices_temp)

            # indices of the new vertices
            index_temp=index[np.logical_not(flag)];


            # store the new indices and the vertices in the main list
            verticesIndex_new[numVertices:(numVertices+numVertices_temp)]=index_temp;
            vertices_new[numVertices:(numVertices+numVertices_temp),:]=vertices_temp;



            #set up the new indeices for the triangles
            triangles_new[(i)*4:(i+1)*4,:]=np.array( [[index[0],index[3],index[5]],
                                               [index[3],index[1],index[4]],
                                               [index[4],index[2],index[5]],
                                               [index[5],index[3],index[4]]]);


            #sum up the number of vertices
            numVertices=numVertices+numVertices_temp;

            #==================================================================
            # housekeeping done!
            #==================================================================

        #replace the old grid with the new grid
        vertices=vertices_new
        triangles=triangles_new

    return vertices, triangles


def vertices_cart2sphere(vertices):
    #transform the coordinates of the vertices from cartesian to spherical


    #check for the right dimensions
    if vertices.ndim==1:
        vertices=np.reshape(vertices,(1,len(vertices)))

    if np.ndim(vertices) !=2:
        raise Exception('vertices must be an N x 3  array');


    r=np.sqrt(np.sum(vertices**2,axis=1))

    theta=np.arccos(vertices[:,2]/r)

    phi=np.arctan2(vertices[:,1],vertices[:,0])

    vertices_sph=np.array([r,theta,phi]).transpose()

    return vertices_sph




def rotate_vertices(vertices,alpha,beta,gamma):
     #rotate the coordinates of the vertices by alpha,beta and gamma
     #IMPORTANT alpha,beta and gamma are in radian

    #check for the right dimensions
    if vertices.ndim==1:
        vertices=np.reshape(vertices,(1,len(vertices)))

    if np.ndim(vertices) !=2:
        raise Exception('vertices must be an N x 3  array');


    # rotation matrix elements
    a11=np.cos(gamma)*np.cos(beta)*np.cos(alpha)-np.sin(gamma)*np.sin(alpha)
    a12=np.cos(gamma)*np.cos(beta)*np.sin(alpha)+np.sin(gamma)*np.cos(alpha)
    a13=-np.cos(gamma)*np.sin(beta)

    a21=-np.sin(gamma)*np.cos(beta)*np.cos(alpha)-np.cos(gamma)*np.sin(alpha)
    a22=-np.sin(gamma)*np.cos(beta)*np.sin(alpha)+np.cos(gamma)*np.cos(alpha)
    a23=np.sin(gamma)*np.sin(beta)

    a31=np.sin(beta)*np.cos(alpha)
    a32=np.sin(beta)*np.sin(alpha)
    a33=np.cos(beta)


    #non rotated coordinates
    x=vertices[:,0]
    y=vertices[:,1]
    z=vertices[:,2]


    #rotate x-coordinate
    x_rot=a11*x+a21*y+a31*z
    y_rot=a12*x+a22*y+a32*z
    z_rot=a13*x+a23*y+a33*z

    vertices_rot=np.array([x_rot,y_rot,z_rot]).transpose()

    return vertices_rot



def interpolation_weights_icosahedric_grid(vertices,D):

    #Function that calculates the interpolation weights for interpolation on
    #a facet(triangle) of an icosahedric grid using a barycentric interpolation
    #
    #vertices are an M x 3 array with the cartesian coordinates of the vertices
    #D is the interpolation point


    #Firn the three most similar vertices by calculating the dot product of D
    # with each vertex, the vertices of the three highest dotproducts span the
    # triangle, in which the interpolation point D lies.
    similarity=np.zeros(np.size(vertices,axis=0))
    for i in range(0,np.size(vertices,axis=1)):

       similarity=vertices[:,i]*D[i]+similarity;



    #get the three most similar vertices
    idx=np.argsort(similarity);

    #reverse it
    idx=idx[::-1]

    #we need only the first three vertices
    idx=idx[0:3]

    # get the weights for the interpolation

    #give each corner an index
    tip_index=np.array([0,1,2]);

    #Allocate
    w=np.zeros(3);

    #loop over the corners of th triangle, to calculate the three area of the
    #triangle for the barycentric interpolation.
    for i in range(0,3):

        #cyclic permutation of the corner index
        js=np.roll(tip_index,i);


        B=vertices[idx[js[1]],:];
        C=vertices[idx[js[2]],:];

        #span the subtriangle
        BC=np.linalg.norm(C-B);
        DB=np.linalg.norm(B-D);
        DC=np.linalg.norm(C-D);

        #Use Heron's formula to calculate area of triangle
        s=(BC+DB+DC)/2;
        w[i]=np.sqrt(np.abs(s*(s-BC)*(s-DB)*(s-DC)));


    #the weights
    w=w/np.sum(w);

    return w,idx


def rotater_mat(eta):
    #Important eta has to be in rad!!
    #Stokes rotation matrix


    P=np.zeros((4,4))
    P[0,0]=1
    P[1,1]=np.cos(2*eta)
    P[2,1]=np.sin(2*eta)
    P[1,2]=-P[2,1]
    P[2,2]=P[1,1]
    P[3,3]=1

    return P





def rotatepoles(vertices, ext_mat_array,alpha):
    #Important alpha has to be in rad!!
        
    ext_mat_array_local=ext_mat_array.copy()
    
    
    #Top pole
    Tp=np.array([0.,0.,1.])
    
    #bottom pole
    Bp=np.array([0.,0.,-1.])
    
    
    #get the vertex of the top pole
    [idx_Tp,dist1]=find_nearest_Nvertices(vertices, Tp)
    
    #get the vertex of the bottom pole
    [idx_Bp,dist2]=find_nearest_Nvertices(vertices, Bp)

    if dist1<1e-7 and dist2<1e-7:

        #for the extinction matrix, the problem is similar to rotate the stokes vector
        
        
        # pole matrices
        ext_mat_array_poles=np.vstack((ext_mat_array_local[idx_Tp,:,:],ext_mat_array_local[idx_Bp,:,:]))    

    
        #polarzer matrix
        P1=rotater_mat(alpha)
        P2=rotater_mat(-alpha)
    
        
        ext_mat_array_poles=np.dot(ext_mat_array_poles,P1)   
        BA = np.transpose(np.dot(np.transpose(ext_mat_array_poles,(0,2,1)),P2.T),(0,2,1))
        ext_mat_array_poles=BA
     
    
        ext_mat_array_local[idx_Tp,:,:]=ext_mat_array_poles[0,:,:]
        ext_mat_array_local[idx_Bp,:,:]=ext_mat_array_poles[1,:,:]
    
    
    return ext_mat_array_local     
