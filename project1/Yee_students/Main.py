import Yee_FDTD
import mesh
import matplotlib.pyplot as pt
import numpy as np
import scipy.constants as const

                            #####
                            #Yee#
                            #####

#general remark: time and related variables, the electric field and the sigmas 
# have been rescaled (see page 36 of the syllabus for more details) 


#Initialization of the mesh with arguments size in the x-direction, size in 
#the y_direction, number of grid cells in x-direction and number of grid cells in y-direction
mesh1=mesh.mesh(5, 5, 150,150,plot=False)

#%%

#Initialization of the Yee simulation, specifying the mesh, the duration,
#the Courant number and the preferred boundary condition (only PBC is implemented)
yee1=Yee_FDTD.Yee_FDTD(mesh1,20,Courant_factor=1,boundary_condition='PBC')

print(yee1.timestep)

#Addition of a dielectric medium (rectangle), with arguments relative permittivity, lower x-bound,
#upper x-bound, lower y-bound and upper y-bound 
yee1.add_dielectric(3,1,2,3,0,5)

#Addition of a PML layer, also adapts the mesh, with arguments the number of layers,
#kappa_max and the the value of m
yee1.add_PML(30,5,m=4)

#Addition of the source, only one source is possible, with arguments the x-coord, 
#the y_coord, source amplitude, source sigma (width of the gaussian), source type (Gaussian and 
#Gaussian modulated are possible), central frequency (only applicable for Gaussian Modulated)
x_source=1.5
y_source=2.5
yee1.add_source(x_source,y_source,2,0.25,source_type='Gaussian', central_frequency=0)


#Addition of an observation point with arguments x-coord and y-coord
x_ob=2.5
y_ob=2.5
yee1.add_observation(x_ob,y_ob)

yee1.run()

#returns a list, for each observation point it contains an array with the value 
#of the electric field at every iteration
output=yee1.output

output = list(output[0])
print(np.shape(output))
pt.plot(range(len(output)), output)
pt.show()




