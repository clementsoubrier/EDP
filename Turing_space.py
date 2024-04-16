import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import cv2
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.patches import Polygon
import matplotlib as mpl



@njit
def compute_slice(d, step, a_min, a_max, b_min, b_max):
    dim_1 = int((a_max - a_min)/step)
    dim_2 = int((b_max - b_min)/step)
    res = np.zeros((dim_1, dim_2),dtype = np.bool_)
    for i,a in enumerate(np.linspace(a_min, a_max, dim_1)):
        for j,b in enumerate(np.linspace(b_min, b_max, dim_2)):
            us = a+b
            vs = b / (us**2)
            fu = -1 + 2*us*vs
            fv = us**2
            gu = -2.*us*vs
            gv = -us**2

            res[i,j] = (fu+gv<0) and (fu*gv-fv*gu>0) and (d*fu+gv>0) and ((d*fu+gv)**2 - 4*d*(fu*gv-fv*gu)>0)
    return res

def main():
    step = 0.001 
    sclices = 20
    a_min = 0.1
    a_max = 10
    b_min = 0.1
    b_max = 10
    d_max = 25
    d_min = 5
    
    ax = plt.axes(projection='3d')
    
    First = True
    for d in np.linspace(d_min, d_max, sclices):
        
        if First:
            slice = compute_slice(d, step, a_min, a_max, b_min, b_max)
            contours, _= cv2.findContours(slice.astype(np.uint8), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
            if len(contours)>=1:
                First = False
                contour = np.squeeze(contours[-1]).astype(np.float32)
                contour = np.append(contour, np.expand_dims(contour[0], axis=0), axis=0)
                
                contour[:,0] = a_min + contour[:,0]*step
                contour[:,1] = b_min + contour[:,1]*step
                old_d = d
                old_contour  = contour
        else:        
            slice = compute_slice(d, step, a_min, a_max, b_min, b_max)
            
            contours, _= cv2.findContours(slice.astype(np.uint8), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
            if len(contours)>=1:
                contour = np.squeeze(contours[-1]).astype(np.float32)
                contour = np.append(contour, np.expand_dims(contour[0], axis=0), axis=0)
                
                contour[:,0] = a_min + contour[:,0]*step
                contour[:,1] = b_min + contour[:,1]*step
                ax.plot3D(contour[:,0], contour[:,1], d * np.ones(len(contour[:,1])),c="k")
                
                cmap=mpl.colormaps['viridis']
                # poly = Polygon(contour, color = cmap((d-d_min)/(d_max - d_min)), 
                #     alpha = 0.2)
                # ax.add_patch(poly)
                # art3d.pathpatch_2d_to_3d(poly, z=d)
                
                old_contour_len = len (old_contour)
                new_contour_len = len (contour)
                val = np.linspace(0,new_contour_len-1, old_contour_len).astype(int)
                new_contour = contour[val]
                xs_3d = np.vstack((old_contour[:,0].T,new_contour[:,0].T))
                ys_3d = np.vstack((old_contour[:,1].T,new_contour[:,1].T))
                zs_3d = np.zeros((2,old_contour_len))
                zs_3d[0,:] = old_d
                zs_3d[1,:] = d
                ax.plot_surface(xs_3d, ys_3d, zs_3d,  color = cmap((d-d_min)/(d_max - d_min)), lw=0.5, rstride=1,
                        cstride=1, alpha=0.7, edgecolor='none',norm=mpl.colors.PowerNorm(gamma=1.5)
                        )
                
                old_d = d
                old_contour  = contour
    ax.set_xlabel('a')
    ax.set_ylabel('b')
    ax.set_zlabel('d')
    plt.show()
        
        

        
if __name__ == '__main__':
    main()

