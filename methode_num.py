# -*- coding: utf-8 -*-

import numpy as np
import numpy.linalg as lg

# --- Méthodes numériques de résolution d'équations différentielles --- #



# --- Méthodes à un pas --- #


# --- Méthode d'Euler --- #

def step_euler(y, t, h, f):
    """ La fonction "step_euler" prend 4 arguments :
        - y : tableau representant les points d'ordonnées
        - t : tableau representant les points d'abscisses
        - h : tableau contenant le pas 
        - f : fonction
        Retourne 
    """
    res = y
    k = 0

    for i in range(0, y.shape[0]):
        res[i] = y[i] + h[i] * f[i](t, y[i])
        
    return res


# --- Méthode du point milieu --- #

def step_point_milieu(y,t,h,f):
    """ La fonction "step_point_milieu" prend 4 arguments :
        - y : tableau representant les points d'ordonnées
        - t : tableau representant les points d'abscisses
        - h : tableau contenant le pas 
        - f : fonction
        Retourne 
    """
    n = np.shape(y)[0]
    a = np.zeros([n,1])
    a.shape=(n)
    b = np.zeros([n,1])
    for i in range(n):
        a[i] = y[i] + (h/2) * f[i](y,t)
    for i in range(n):
        b[i] = f[i](a,t+h/2)
    for i in range(n):
        a[i] = y[i] + h*b[i]
    return a


# --- Méthode de Heun --- #

def step_heun(y,t,h,f):
    """ La fonction "step_heun" prend 4 arguments :
        - y : tableau representant les points d'ordonnées
        - t : tableau representant les points d'abscisses
        - h : tableau contenant le pas 
        - f : fonction
        Retourne 
    """
    n = np.shape(y)[0]
    a = np.zeros([n,1])
    a.shape=(n)
    p1 = np.zeros([n,1])
    p1.shape=(n)
    p2 = np.zeros([n,1])
    p2.shape=(n)
    for i in range(n):
        p1[i] = f[i](y,t)
    for i in range(n):
        a[i]  = y[i] + h*p1[i]
    for i in range(n):
        p2[i] = f[i](a,t+h)
    for i in range(n):
        a[i] = y[i] + (h/2) * (p1[i] + p2[i])
    
    return a


# --- Méthode de Runge-Kutta --- #


def step_runge_kutta(y,t,h,f):
    """ La fonction "step_runge_kutta" prend 4 arguments :
        - y : tableau representant les points d'ordonnées
        - t : tableau representant les points d'abscisses
        - h : tableau contenant le pas 
        - f : fonction
        Retourne 
    """
    n = np.shape(y)[0]
    p1 = np.zeros([n,1])
    p1.shape=(n)
    p2 = np.zeros([n,1])
    p2.shape=(n)
    p3 = np.zeros([n,1])
    p3.shape=(n)
    p4 = np.zeros([n,1])
    p4.shape=(n)
    a = np.zeros([n,1])
    a.shape=(n)
    for i in range(n):
        p1[i] = f[i](y,t)
    
    for i in range(n):
        a[i] = y[i] + h/2 * p1[i]
    
    for i in range(n):
        p2[i] = f[i](a,t+ h/2)
    
    for i in range(n):
        a[i] = y[i] + h/2 * p2[i]
    for i in range(n):
        p3[i] = f[i](a,t+ h/2)
    
    for i in range(n):
        a[i] = y[i] + h * p3[i]
    for i in range(n):
        p4[i] = f[i](a,t+ h)
    
    for i in range(n):
        a[i] = y[i] + (1./6.) * h * (p1[i] + 2*p2[i] + 2*p3[i] + p4[i])
    
    return a



# --- N pas de taille h --- #


def meth_n_step(y0, t0, N, h, f, step_meth):
    """ La fonction "meth_n_step" prend 7 arguments :
        - y0 :
        - t0 :
        - N : 
        - h : tableau contenant le pas 
        - f : fonction
        - step_meth :
        Retourne 
    """
    y = np.zeros([N, y0.size])
    t = t0
    y[0,:] = y0
    
    for i in range(1,N):
        y[i,:] = step_meth(y[i-1,:], t, h, f)
        t = t + h
    
    return y


# --- Epsilon --- #

# fonction renvoyant une seule valeur

def meth_epsilon(y0,t0,tf,eps,f,meth):

    flag = 0
    error = eps + 1 #on met l'erreur relative au dessus de epsilon pour rentrer dans la boucle
    N = FIRST_N
    h = (tf-t0) / float(N)
    yf_old = meth_n_step(y0, t0, N, h, f, meth)
    
    while (error > eps and flag < MAX_STEP):
        N *= 2
        h /= 2
        yf = meth_n_step(y0, t0, N, h, f, meth)
        error = np.linalg.norm(yf - yf_old)
        yf_old = yf
        flag += 1
    
    if flag == MAX_STEP:
        print "More steps are needed"

    return yf





def main():
    meth_epsilon()
    

if __name__ ==  '__main__':
    main()
