# -*- coding: utf-8 -*-
"""
Created Jun14 2021

@author: H.P. Wang
github:  https://github.com/hpwang87
"""
import sympy as sp
import numpy as np
import scipy.io as sio
import os
import matplotlib.pyplot as plt


def flow2D(tvec, xvec, yvec):
    """
    Parameters
    ----------
    tvec : time 
    xvec : x nodes
    yvec : y nodes

    Returns
    -------
    u, v, p
    """
    x, y, t = sp.symbols('x y t')
    u = -sp.cos(x)*sp.sin(y)*sp.exp(-2.0*t)
    u_f = sp.lambdify([x, y, t], u, 'numpy')
    v = sp.sin(x)*sp.cos(y)*sp.exp(-2.0*t)
    v_f = sp.lambdify([x, y, t], v, 'numpy')    
    p = -0.25*(sp.cos(2.0*x)+sp.cos(2.0*y))*sp.exp(-4.0*t)
    p_f = sp.lambdify([x, y, t], p, 'numpy')       
    
    ux = sp.diff(u, x)
    ux_f = sp.lambdify([x, y, t], ux, 'numpy')        
    uy = sp.diff(u, y)
    uy_f = sp.lambdify([x, y, t], uy, 'numpy')       
    ut = sp.diff(u, t)
    ut_f = sp.lambdify([x, y, t], ut, 'numpy')    
    vx = sp.diff(v, x)
    vx_f = sp.lambdify([x, y, t], vx, 'numpy')    
    vy = sp.diff(v, y)
    vy_f = sp.lambdify([x, y, t], vy, 'numpy')    
    vt = sp.diff(v, t)
    vt_f = sp.lambdify([x, y, t], vt, 'numpy')    
    px = sp.diff(p, x)
    px_f = sp.lambdify([x, y, t], px, 'numpy')    
    py = sp.diff(p, y)
    py_f = sp.lambdify([x, y, t], py, 'numpy')    
    pt = sp.diff(p, t)
    pt_f = sp.lambdify([x, y, t], pt, 'numpy')   
    
    uxx = sp.diff(ux, x)
    uxx_f = sp.lambdify([x, y, t], uxx, 'numpy')      
    uyy = sp.diff(uy, y)
    uyy_f = sp.lambdify([x, y, t], uyy, 'numpy')      
    vxx = sp.diff(vx, x)
    vxx_f = sp.lambdify([x, y, t], vxx, 'numpy')      
    vyy = sp.diff(vy, y)    
    vyy_f = sp.lambdify([x, y, t], vyy, 'numpy')  
    
    u_val = u_f(xvec, yvec, tvec)
    v_val = v_f(xvec, yvec, tvec) 
    p_val = p_f(xvec, yvec, tvec)
    
    ux_val = ux_f(xvec, yvec, tvec)
    uy_val = uy_f(xvec, yvec, tvec)
    ut_val = ut_f(xvec, yvec, tvec)
    vx_val = vx_f(xvec, yvec, tvec) 
    vy_val = vy_f(xvec, yvec, tvec)
    vt_val = vt_f(xvec, yvec, tvec)
    px_val = px_f(xvec, yvec, tvec)
    py_val = py_f(xvec, yvec, tvec)  
    pt_val = pt_f(xvec, yvec, tvec)  
    
    uxx_val = uxx_f(xvec, yvec, tvec)
    uyy_val = uyy_f(xvec, yvec, tvec)
    vxx_val = vxx_f(xvec, yvec, tvec)
    vyy_val = vyy_f(xvec, yvec, tvec)     
    
    return {'u':u_val, 'v':v_val, 'p':p_val,
            'ux':ux_val, 'uy':uy_val, 'ut':ut_val,
            'vx':vx_val, 'vy':vy_val, 'vt':vt_val,
            'px':px_val, 'py':py_val, 'pt':pt_val,
            'uxx':uxx_val, 'uyy':uyy_val,
            'vxx':vxx_val, 'vyy':vyy_val}


"""
for testing
"""
if __name__ == "__main__":
    lmin = 0
    lmax = 2.0*np.pi
    lnum = 64
    tmin = 0
    tmax = 2
    tnum = 64
    xvec = np.linspace(lmin, lmax, lnum)
    yvec = np.linspace(lmin, lmax, lnum)
    tvec = np.linspace(tmin, tmax, tnum)
    
    xmesh,ymesh,tmesh = np.meshgrid(xvec,\
                                    yvec,\
                                    tvec,indexing='xy') 
    
    uvp_ture = flow2D(tmesh, xmesh, ymesh)
    
    u = uvp_ture['u']
    v = uvp_ture['v']
    p = uvp_ture['p']
    ux = uvp_ture['ux']
    uy = uvp_ture['uy']
    ut = uvp_ture['ut']
    vx = uvp_ture['vx']
    vy = uvp_ture['vy']  
    vt = uvp_ture['vt']    
    px = uvp_ture['px']
    py = uvp_ture['py']   
    uxx = uvp_ture['uxx']
    uyy = uvp_ture['uyy']   
    vxx = uvp_ture['vxx']
    vyy = uvp_ture['vyy']  
    
    Re = 1
    e1 = ut+u*ux+v*uy+px-1/Re*(uxx+uyy)
    e2 = vt+u*vx+v*vy+py-1/Re*(vxx+vyy)
    e3 = ux+vy
    
    plt.figure()
    ax1 = plt.subplot(2,2,1)
    ax2 = plt.subplot(2,2,2)
    p1 = ax1.pcolor(xmesh[:,:,0],ymesh[:,:,0],u[:,:,10],cmap='RdYlGn_r', 
                    shading='auto',vmin=-1, vmax=1)
    ax1.quiver(xmesh[:,:,0],ymesh[:,:,0], u[:,:,10], v[:,:,10])
    ax1.set_title('u',fontsize=12,color='r')
    ax1.axis('square')
    p2 = ax2.pcolor(xmesh[:,:,0],ymesh[:,:,0],v[:,:,10],cmap='RdYlGn_r', 
                    shading='auto', vmin=-1, vmax=1)
    ax2.quiver(xmesh[:,:,0],ymesh[:,:,0], u[:,:,10], v[:,:,10])
    ax2.set_title('v',fontsize=12,color='r')
    ax2.axis('square')    
    plt.colorbar(p1, ax=ax1)
    plt.colorbar(p2, ax=ax2)
    
    ax3 = plt.subplot(2,2,3)
    ax4 = plt.subplot(2,2,4)  
    p3 = ax3.pcolor(xmesh[:,:,0],ymesh[:,:,0],e1[:,:,10],cmap='RdYlGn_r', 
                    shading='auto', vmin=-1e-12, vmax=1e-12)
    ax3.set_title('e1',fontsize=12,color='r')
    ax3.axis('square')
    p4 = ax4.pcolor(xmesh[:,:,0],ymesh[:,:,0],e3[:,:,10],cmap='RdYlGn_r', 
                    shading='auto', vmin=-1e-12, vmax=1e-12)
    ax4.set_title('e2',fontsize=12,color='r')
    ax4.axis('square')    
    plt.colorbar(p3, ax=ax3)
    plt.colorbar(p4, ax=ax4)
    plt.show()
    
    # 保存数据
    sizU = u.shape
    xmesh = xmesh[:,:,0]
    ymesh = ymesh[:,:,0]
    all_data_u = u
    all_data_v = v
    all_data_p = p
    filepath = './predict_results'
    filename = 'data_Flow2D_exact.mat'
    freq = 1/(tvec[1]-tvec[0])
    mask = np.zeros(sizU)
    sio.savemat(os.path.join(filepath, filename),\
                {'xmesh':xmesh,\
                 'ymesh':ymesh,\
                 'all_data_u':all_data_u,\
                 'all_data_v':all_data_v,\
                 'all_data_p':all_data_p,\
                 'freq':freq,\
                 'mask':mask})
    
    
    
    