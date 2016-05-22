# -*- coding: utf-8 -*-

#http://www.tangentex.com/PenduleMaple.htm

import numpy as np
import math
import matplotlib as ma
ma.use('Agg')
import matplotlib.pyplot as plt
import classe

g = 9.80665             #acceleration de la pesanteur
L = 1.0                 #longueur du pendule
m = 1                   # masse du pendule

#voir http://mpsi.pauleluard.free.fr/IMG/pdf/pendule_simple.pdf

def f(X, t):
    return np.array([X[1], -g/L*np.sin(X[0])])

def aff_courbe_freq_oscillation(theta0, tf):
    """ La fonction "aff_courbe_freq_oscillation" prend 2 arguments :
        - theta0 : liste de valeurs theta0
        - tf : borne max des abscisses
        Affiche la courbe de la fréquence d'oscillation du système en fonction de theta
    """
    plt.title('Oscillations')
    plt.xlabel(r'$t$',fontsize=26)
    plt.ylabel(r'$\theta(t)$',fontsize=20)
    
    for i in (theta0):
        s = classe.pCauchy(0, [i, 0.0], f)
        tabAngle = (s.meth_epsilon(tf, 0.01, s.step_runge_kutta))
        
        n = np.shape(tabAngle)[0]
        t = np.linspace(0, tf, n, endpoint=False)
        plt.plot(t, tabAngle[:,0])
        
    plt.show()
    plt.savefig("freq_oscillation.png")
    plt.close()



def calc_freq_oscillation(theta0, tf):
    """ La fonction "calc_freq_oscillation" prend 2 arguments :
        - theta0 : liste de valeurs theta0
        - tf : borne max des abscisses
        Retourne la liste des fréquence d'oscillation de chacun des systèmes si possible, retourne une liste vide sinon.
    """
    tabFreq = []
    
    for i in (theta0):
        s = classe.pCauchy(0, [i, 0.0], f)
        tabAngle = (s.meth_epsilon(tf, 0.01, s.step_runge_kutta))
    
        n = np.shape(tabAngle)[0]

        iMax = 0
        iMin = 0
        j=1
        while (j < n and tabAngle[j,0] >= tabAngle[iMax,0]):
            iMax = float(j)
            j=j+1
        if (j==n):
            print("La fréquence n'a pas pu être calculée, veuillez augmenter tf.")
            return [];
        j=1
        while (j < n and tabAngle[j,0] <= tabAngle[iMin,0]):
            iMin = float(j)
            j=j+1
        if (j==n):
            print("La fréquence n'a pas pu être calculée, veuillez augmenter tf.")
            return; 

        tabFreq+=[ 2.0*abs(iMin-iMax) / n * tf]
    
    return tabFreq
        
        

    
    

def pendule_un_maillon():
    tf=6
    theta0 = np.pi/2
    
    s = classe.pCauchy(0, [theta0, 0.0], f)
    res = s.meth_epsilon(tf, 0.01, s.step_runge_kutta)
    #s.aff_courbe_eq_diff(tf, 0.01, "pend")

    
    aff_courbe_freq_oscillation([theta0], tf)
    calc_freq_oscillation([theta0], tf)
    
    theta0 = [np.pi/3, np.pi/4, -np.pi/2, -np.pi/4, -np.pi/1.3]
    aff_courbe_freq_oscillation(theta0, tf)
    tabFreq = calc_freq_oscillation(theta0, tf)
    
    print( tabFreq, np.sqrt(g/L))

    



if __name__ ==  '__main__':
    pendule_un_maillon()
    
    
    #        - tabAngle : tableau de couple angle theta / vitesse angulaire