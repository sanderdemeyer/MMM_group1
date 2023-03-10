import numpy as np
import matplotlib.pyplot as pt

class mesh:

#This class creates a uniform mesh
#class can be extended to include local refinements.

################
#Initialization#
################

#During the intialization the size of the relevant domain is specified, PMLs not included.
#The dimensions are expressed in m
#n_cells_x and n_cells_y define the number of cells in the x and y direction

########################
#Attributes of the mesh#
########################

#x_size: length in the x-direction of the relevant simulation domain
#y_size: length in the y-direction of the relevant simulation domain
#n_cells_x: number of cells in the x-direction
#n_cells_y: number of cells in the y-direction
#primary_x_vector: x_values for which there is a node in the primary grid of E_z; x_i
#primary_y_vector: y_values for which there is a node in the primary grid of E_z; y_j
#primary_x_delta: distance between 2 subsequent x-nodes in the primary grid of E_z; delta x_i=x_i-x_i-1
#primary_y_delta: distance between 2 subsequent y-nodes in the primary grid of E_z; delta y_i=y_i-y_i-1
#x_step: length of a cell in the x_dimension of the primary grid of E_z
#y_step: length of a cell in the y_dimension of the primary grid of E_z
#grid_ez: contains the coordinates of the grid for E_z
#grid_hy: contains the coordinates of the grid for H_y
#grid_hx: contains the coordinates of the grid for H_x
#dual_x_delta: distance between 2 subsequent x-nodes in the dual grid of H_x and H_y; delta x_i*=(x_i+x_i-1)/2
#dual_y_delta: distance between 2 subsequent y-nodes in the dual grid of H_x and H_y; delta y_j*=(y_j+y_j-1)/2

    
    
    
    def __init__(self, x_size, y_size, n_cells_x,n_cells_y, plot=False):
        self.x_size=x_size
        self.y_size=y_size
        
        self.n_cells_x=n_cells_x
        self.n_cells_y=n_cells_y
        
        self.primary_x_vector=np.linspace(0,self.x_size,self.n_cells_x)
        self.primary_y_vector=np.linspace(0,self.y_size,self.n_cells_y)
        
        self.primary_x_delta=self.primary_x_vector[1:]-self.primary_x_vector[0:-1]
        self.primary_y_delta=self.primary_y_vector[1:]-self.primary_y_vector[0:-1]
        
        self.y_step=self.primary_y_delta[0]
        self.x_step=self.primary_x_delta[0]
        
        self.x_vector_ez=self.primary_x_vector[0:-1]
        self.y_vector_ez=self.primary_y_vector[0:-1]
        
        self.grid_ez=np.meshgrid(self.x_vector_ez,self.y_vector_ez,indexing='ij')
        
        self.dual_x_delta=np.concatenate((np.array([self.primary_x_delta[0]]),(self.primary_x_delta[0:-1]+self.primary_x_delta[1:])/2))
        self.dual_y_delta=np.concatenate((np.array([self.primary_y_delta[0]]),(self.primary_y_delta[0:-1]+self.primary_y_delta[1:])/2))
        
        self.x_vector_hy=np.cumsum(np.append(self.primary_x_delta[0]/2,self.dual_x_delta[1:]))
        self.y_vector_hy=self.primary_y_vector[0:-1]
        
        self.x_vector_hx=self.primary_x_vector[0:-1]
        self.y_vector_hx=np.cumsum(np.append(self.primary_y_delta[0]/2,self.dual_y_delta[1:]))
        
        self.grid_hx=np.meshgrid(self.x_vector_hx,self.y_vector_hx,indexing='ij')
        self.grid_hy=np.meshgrid(self.x_vector_hy,self.y_vector_hy,indexing='ij')
        
        if plot:
            self.plot_mesh('Plot of the uniform grid')
                    

    
    def plot_mesh(self, title):
        
        shape=np.shape(self.grid_ez[0])
        X=np.zeros(shape[0]*shape[1])
        Y=np.zeros(shape[0]*shape[1])
        ct=0
        
        for i in range (0,shape[0]):
            for j in range (0,shape[1]):
                X[ct]=self.grid_ez[0][i,j]
                Y[ct]=self.grid_ez[1][i,j]
                ct+=1
                
        shape=np.shape(self.grid_hx[0])
        XX=np.zeros(shape[0]*shape[1])
        YY=np.zeros(shape[0]*shape[1])
        ct=0
        
        for i in range (0,shape[0]):
            for j in range (0,shape[1]):
                XX[ct]=self.grid_hx[0][i,j]
                YY[ct]=self.grid_hx[1][i,j]
                ct+=1
        
        shape=np.shape(self.grid_hy[0])
        XXX=np.zeros(shape[0]*shape[1])
        YYY=np.zeros(shape[0]*shape[1])
        ct=0
        
        for i in range (0,shape[0]):
            for j in range (0,shape[1]):
                XXX[ct]=self.grid_hy[0][i,j]
                YYY[ct]=self.grid_hy[1][i,j]
                ct+=1
        
        pt.figure()
        pt.scatter(X,Y, s=5, color='black')
        pt.scatter(XX,YY,s=4, marker='>',color='red')
        pt.scatter(XXX,YYY,s=4,marker='^',color='blue')
        pt.title(str(title))
        pt.xlabel('Distance in the x-direction (m)')
        pt.ylabel('Distance in the y-direction (m)')
        pt.legend(['$E_z$','$H_x$','$H_y$'], loc='upper right', fontsize='large')
       
       
    def add_PML(self, N_PML,plot=False):
        
        self.x_size=self.x_size+2*N_PML*self.x_step
        self.y_size=self.y_size+2*N_PML*self.y_step
        
        self.n_cells_x=self.n_cells_x+2*N_PML
        self.n_cells_y=self.n_cells_y+2*N_PML
        
        addfrontx=(np.arange(N_PML)-N_PML)*self.x_step+self.primary_x_vector[0]
        addbackx=(np.arange(N_PML)+1)*self.x_step+self.primary_x_vector[-1]
        addfronty=(np.arange(N_PML)-N_PML)*self.y_step+self.primary_y_vector[0]
        addbacky=(np.arange(N_PML)+1)*self.y_step+self.primary_y_vector[-1]
        
        self.primary_x_vector=np.concatenate((addfrontx,self.primary_x_vector,addbackx))
        self.primary_y_vector=np.concatenate((addfronty,self.primary_y_vector,addbacky))
        
        self.primary_x_delta=self.primary_x_vector[1:]-self.primary_x_vector[0:-1]
        self.primary_y_delta=self.primary_y_vector[1:]-self.primary_y_vector[0:-1]
        
        self.x_vector_ez=self.primary_x_vector[0:-1]
        self.y_vector_ez=self.primary_y_vector[0:-1]
        
        self.grid_ez=np.meshgrid(self.x_vector_ez,self.y_vector_ez,indexing='ij')
        
        self.dual_x_delta=np.concatenate((np.array([self.primary_x_delta[0]]),(self.primary_x_delta[0:-1]+self.primary_x_delta[1:])/2))
        self.dual_y_delta=np.concatenate((np.array([self.primary_y_delta[0]]),(self.primary_y_delta[0:-1]+self.primary_y_delta[1:])/2))
        
        self.x_vector_hy=np.cumsum(np.append(self.primary_x_vector[0]+self.x_step/2,self.dual_x_delta[1:]))
        self.y_vector_hy=self.primary_y_vector[0:-1]
        
        self.x_vector_hx=self.primary_x_vector[0:-1]
        self.y_vector_hx=np.cumsum(np.append(self.primary_y_vector[0]+self.y_step/2,self.dual_y_delta[1:]))
        
        self.grid_hx=np.meshgrid(self.x_vector_hx,self.y_vector_hx,indexing='ij')
        self.grid_hy=np.meshgrid(self.x_vector_hy,self.y_vector_hy,indexing='ij')
        if plot:
            self.plot_mesh('Plot after addition of the PML layers') 
            
