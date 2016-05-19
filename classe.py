# -*- coding: utf-8 -*-

import numpy as np
import numpy.linalg as lg

class pCauchy(object):
    def __init__(self, t0, y0, f):
        self.t0=t0
        self.y0=y0
        self.f = f
        
    # --- Méthodes à un pas --- #
    
    def step_euler(self, y, t, h):
        """ La fonction "step_euler" prend 4 arguments :
            - self : classe représentant le problème de Cauchy
            - y : point d'ordonnée
            - t : point d'abscisse
            - h : pas 
            Retourne 
        """
        #res = y
        #for i in range(0, y.shape[0]):
        resf= self.f(y, t)
        hxf = [(h*i) for i in resf]
        res = y + hxf #y + h * self.f(y, t)
        return res

    # --- Méthode du point milieu --- #
    
    def step_point_milieu(self, y,t,h):
        """ La fonction "step_point_milieu" prend 4 arguments :
            - self : classe représentant le problème de Cauchy
            - y : point d'ordonnée
            - t : point d'abscisse
            - h : pas 
            Retourne 
        """
        n = np.shape(y)[0]
        a = np.zeros([n,1])
        a.shape=(n)
        b = np.zeros([n,1])
        for i in range(n):
            a[i] = y[i] + (h/2) * self.f(y,t)[i]
        #for i in range(n):
        b = self.f(a,t+h/2)
        for i in range(n):
            a[i] = y[i] + h*b[i]
        return a
    
    # --- Méthode de Heun --- #
    
    def step_heun(self, y,t,h):
        """ La fonction "step_heun" prend 4 arguments :
            - self : classe représentant le problème de Cauchy
            - y : point d'ordonnée
            - t : point d'abscisse
            - h : pas 
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
            p1[i] = self.f[i](y,t)
        for i in range(n):
            a[i]  = y[i] + h*p1[i]
        for i in range(n):
            p2[i] = self.f[i](a,t+h)
        for i in range(n):
            a[i] = y[i] + (h/2) * (p1[i] + p2[i])
        return a
    
    # --- Méthode de Runge-Kutta --- #
    
    def step_runge_kutta(self, y,t,h):
        """ La fonction "step_runge_kutta" prend 4 arguments :
            - self : classe représentant le problème de Cauchy
            - y : point d'ordonnée
            - t : point d'abscisse
            - h : pas 
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
            p1[i] = self.f[i](y,t)
        
        for i in range(n):
            a[i] = y[i] + h/2 * p1[i]
        
        for i in range(n):
            p2[i] = self.f[i](a,t+ h/2)
        
        for i in range(n):
            a[i] = y[i] + h/2 * p2[i]
        for i in range(n):
            p3[i] = self.f[i](a,t+ h/2)
        
        for i in range(n):
            a[i] = y[i] + h * p3[i]
        for i in range(n):
            p4[i] = self.f[i](a,t+ h)
        
        for i in range(n):
            a[i] = y[i] + (1./6.) * h * (p1[i] + 2*p2[i] + 2*p3[i] + p4[i])
        return a
        
    # --- N pas de taille h --- #
    
    def meth_n_step(self, N, h, step_meth):
        """ La fonction "meth_n_step" prend 7 arguments :
            - self : classe représentant le problème de Cauchy
            - N : nombre de pas
            - h : pas 
            - f : fonction de résolution
            - step_meth :
            Retourne 
        """
        if (isinstance(self.y0,int)):
            y = np.zeros([N, 1])
        else:
            y = np.zeros([N, len(self.y0)]) 
        t = self.t0
        y[0,:] = self.y0
        
        for i in range(1,N):
            y[i,:] = step_meth(y[i-1,:], t, h)
            t = t + h
        return y

    # --- Epsilon --- #
    
    # fonction renvoyant une seule valeur
    
    def meth_epsilon(self,tf,eps,f,meth):
        """ La fonction "meth_epsilon" prend 6 arguments :
            - self : classe représentant le problème de Cauchy
            - tf : 
            - eps : erreur maximale
            - f : fonction
            - meth :
            Retourne 
        """
        MAX_STEP = 2**16
        flag = 0
        error = eps + 1 #on met l'erreur relative au dessus de epsilon pour rentrer dans la boucle
        N = 1
        h = (tf-self.t0) / float(N)
        yf_old = self.meth_n_step(N, h, meth)
        
        while (error > eps and flag < MAX_STEP):
            N *= 2
            h /= 2
            yf = self.meth_n_step(N, h, meth)
            error = np.linalg.norm(yf - yf_old)
            yf_old = yf
            flag += 1
        
        if flag == MAX_STEP:
            print "More steps are needed"
        return yf
