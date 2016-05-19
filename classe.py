# -*- coding: utf-8 -*-

import numpy as np
import numpy.linalg as lg

import matplotlib as ma
ma.use('Agg')
import matplotlib.pyplot as plt

class pCauchy(object):
    def __init__(self, t0, y0, f):
        self.t0=t0
        self.y0=y0
        self.f = f

    def champTangente(self, xmin=-10, xmax=10, ymin=-10, ymax=10, pas=1):
        """ La fonction "champTangente" prend 4 arguments :
            - self : classe représentant le problème de Cauchy
            - xmin : abscisse de la borne minimale
            - xmax : abscisse de la borne maximale
            - ymin : ordonnée de la borne minimale
            - ymax : ordonnée de la borne maximale
            - pas : pas 
            Affiche le champ des tangentes de l'équation différentielle
        """
        X = []
        Y = []
        U = []
        V = []
        print(len(V))
        for t in range(xmin,xmax,pas):
            for yt in range(ymin, ymax, pas):
                X.append(t)
                Y.append(yt)
                U.append(1)
                V.append(self.f(yt,t))
        Q = plt.quiver(X,Y,U,V)
        plt.savefig("champ_tangente.png")
        plt.show()
        plt.clf()

    # --- Méthodes à un pas --- #
    
    def step_euler(self, y, t, h):
        """ La fonction "step_euler" prend 4 arguments :
            - self : classe représentant le problème de Cauchy
            - y : ordonnée du point courant
            - t : abscisse du point courant
            - h : pas 
            Retourne une approximation de l'ordonnée du point suivant
        """
        f= self.f(y, t)
        res = y + [(h*i) for i in f] #y + h * self.f(y, t)

        return res

    # --- Méthode du point milieu --- #
    
    def step_point_milieu(self, y,t,h):
        """ La fonction "step_point_milieu" prend 4 arguments :
            - self : classe représentant le problème de Cauchy
            - y : ordonnée du point courant
            - t : abscisse du point courant
            - h : pas 
            Retourne une approximation de l'ordonnée du point suivant
        """
        n = np.shape(y)[0]
        a = np.zeros([n,1])
        a.shape=(n)
        b = np.zeros([n,1])
        for i in range(n):
            a[i] = y[i] + (h/2) * self.f(y,t)[i]
        b = self.f(a,t+h/2)
        for i in range(n):
            a[i] = y[i] + h*b[i]
        return a
    
    # --- Méthode de Heun --- #
    
    def step_heun(self, y,t,h):
        """ La fonction "step_heun" prend 4 arguments :
            - self : classe représentant le problème de Cauchy
            - y : ordonnée du point courant
            - t : abscisse du point courant
            - h : pas 
            Retourne une approximation de l'ordonnée du point suivant
        """
        n = np.shape(y)[0]
        a = np.zeros([n])
        p1 = np.zeros([n])
        p2 = np.zeros([n])
        
        p1 = self.f(y,t)
        for i in range(n):
            a[i]  = y[i] + h*p1[i]

        p2 = self.f(a,t+h)
        for i in range(n):
            a[i] = y[i] + (h/2) * (p1[i] + p2[i])
        return a
    
    # --- Méthode de Runge-Kutta --- #
    
    def step_runge_kutta(self, y,t,h):
        """ La fonction "step_runge_kutta" prend 4 arguments :
            - self : classe représentant le problème de Cauchy
            - y : ordonnée du point courant
            - t : abscisse du point courant
            - h : pas 
            Retourne une approximation de l'ordonnée du point suivant
        """
        n = np.shape(y)[0]
        p1 = np.zeros([n])
        p2 = np.zeros([n])
        p3 = np.zeros([n])
        p4 = np.zeros([n])
        a = np.zeros([n])

        p1 = self.f(y,t)
        for i in range(n):
            a[i] = y[i] + h/2 * p1[i]
        
        p2 = self.f(a,t+ h/2)
        for i in range(n):
            a[i] = y[i] + h/2 * p2[i]

        p3 = self.f(a,t+ h/2)
        for i in range(n):
            a[i] = y[i] + h * p3[i]

        p4 = self.f(a,t+ h)
        for i in range(n):
            a[i] = y[i] + (1./6.) * h * (p1[i] + 2*p2[i] + 2*p3[i] + p4[i])
        return a
        
    # --- N pas de taille h --- #
    
    def meth_n_step(self, N, h, step_meth):
        """ La fonction "meth_n_step" prend 5 arguments :
            - self : classe représentant le problème de Cauchy
            - N : nombre de pas
            - h : pas 
            - f : fonction de résolution
            - step_meth : fonction de résolution pas à pas à utiliser parmi les quatres méthodes implémentées
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
    
    def meth_epsilon(self,tf,eps,meth):
        """ La fonction "meth_epsilon" prend 4 arguments :
            - self : classe représentant le problème de Cauchy
            - tf : borne max des abscisses 
            - eps : erreur maximale admise
            - meth : fonction de résolution pas à pas à utiliser parmi les quatres méthodes implémentées
            Retourne 
        """
        MAX_STEP = 2**4
        flag = 0
        error = eps + 1 #on met l'erreur relative au dessus de epsilon pour rentrer dans la boucle
        N = 2
        h = (tf-self.t0) / float(N)
        yf_old = self.meth_n_step(N, h, meth)
        
        while (error > eps and flag < MAX_STEP):
            N *= 2
            h /= 2
            yf = self.meth_n_step(N, h, meth)
            error = np.linalg.norm(yf[::2] - yf_old)/N
            yf_old = yf
            flag += 1
        
        if flag == MAX_STEP:
            print "More steps are needed"
        return yf
        
    # --- Affichage courbe équation différentielle --- #
            
    def aff_courbe_eq_diff(self, tf, eps = 10E-3):
        """ La fonction  "aff_courbe_eq_diff" prend 3 arguments :
            - self : classe représentant le problème de Cauchy
            - tf : borne max des abscisses
            - eps : erreur maximale admise
            Affiche les courbes de l'équation différentielle calculées à l'aide des 4 méthodes implémentées
        """
        euler = self.meth_epsilon(tf, eps, self.step_euler)
        pm = self.meth_epsilon(tf, eps, self.step_point_milieu)
        heun = self.meth_epsilon(tf, eps, self.step_heun)
        rk = self.meth_epsilon(tf, eps, self.step_runge_kutta)
        
        t_euler = np.linspace(0, tf, np.shape(euler)[0], endpoint=False)
        t_pm = np.linspace(0, tf, np.shape(pm)[0], endpoint=False)
        t_heun = np.linspace(0, tf, np.shape(heun)[0], endpoint=False)
        t_rk = np.linspace(0, tf, np.shape(rk)[0], endpoint=False)
        
        plt.title("Courbes de l'equation differentielle")
        plt.xlabel("x")
        plt.ylabel("y")

        for i in range(np.shape(euler)[1]):
            plt.plot(t_euler, euler[:,i], linewidth=1.0, label="Euler")
            plt.plot(t_pm, pm[:,i], linewidth=1.0, label="Point milieu")
            plt.plot(t_heun, heun[:,i], linewidth=1.0, label="Heun")
            plt.plot(t_rk, rk[:,i], linewidth=1.0, label="Runge-Kutta")
            
            plt.legend(loc='best')
            plt.show()
            plt.savefig("courbes_eq_diff"+str(i)+".png")
            plt.close()


"""

Exemples:

"""

def fex1(yt,t):
    return yt / (1 + t**2)



def exemple1():
    ex1 = pCauchy(0,1,fex1)
    ex1.champTangente()
    


def tests():
    nbPas=10
    pas=0.1
    eps = 10E-2
    
    f = lambda x, t: x/(1-t**2)
    dim1 = pCauchy(0, 1, f)
    
    g = lambda x, t: [-x[1], x[0]]
    dim2 = pCauchy(0, [1, 0], g)
    
    dim1.aff_courbe_eq_diff(nbPas*pas, eps)
    dim2.aff_courbe_eq_diff(nbPas*pas, eps)



if __name__ ==  '__main__':
    exemple1()
    tests()