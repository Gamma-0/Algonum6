# -*- coding: utf-8 -*-

import numpy as np
import math
import matplotlib as ma
ma.use('Agg')
import matplotlib.pyplot as plt
import classe

g = 9.80665             #acceleration de la pesanteur


#http://www.tangentex.com/PenduleMaple.htm
#voir http://mpsi.pauleluard.free.fr/IMG/pdf/pendule_simple.pdf



def aff_courbe_freq_oscillation(pend_1m, tf):
    """ La fonction "aff_courbe_freq_oscillation" prend 2 arguments :
        - pend_1m : liste de classe représentant le pendule à 1 maillons
        - tf : borne max des abscisses
        Affiche la courbe de la fréquence d'oscillation du système en fonction de theta
    """
    plt.title('Oscillations')
    plt.xlabel(r'$t$')
    plt.ylabel(r'$\theta(t)$')
    
    for pend in (pend_1m):
        tabAngle = pend.meth_epsilon(tf, 0.01, pend.step_runge_kutta)
        
        n = np.shape(tabAngle)[0]
        t = np.linspace(0, tf, n, endpoint=False)
        plt.plot(t, tabAngle[:,0])
        
    plt.show()
    plt.savefig("freq_oscillation.png")
    plt.close()



def calc_freq_oscillation(pend_1m, tf):
    """ La fonction "calc_freq_oscillation" prend 2 arguments :
        - pend_1m : liste de classe représentant le pendule à 1 maillons
        - tf : borne max des abscisses
        Retourne la liste des fréquence d'oscillation de chacun des systèmes si possible, retourne une liste vide sinon.
    """
    tabFreq = []
    for pend in (pend_1m):
        tabAngle = (pend.meth_epsilon(tf, 0.01, pend.step_runge_kutta))
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
    L = 1.0                 # longueur de la tige
    f = lambda X, t: np.array([X[1], -g/L*np.sin(X[0])])
    
    tf=6
    theta0 = np.pi/2
    
    pend_1m = [classe.pCauchy(0, [theta0, 0.0], f)]             # y0 : couple angle theta / vitesse angulaire
    #res = s.meth_epsilon(tf, 0.01, s.step_runge_kutta)
    #s.aff_courbe_eq_diff(tf, 0.01, "pend")

    aff_courbe_freq_oscillation(pend_1m, tf)
    calc_freq_oscillation(pend_1m, tf)
    
    theta0 = [np.pi/3, np.pi/4, -np.pi/2, -np.pi/4, -np.pi/1.3]
    for i in (theta0):
        pend_1m += [classe.pCauchy(0, [i, 0.0], f)]
        
    aff_courbe_freq_oscillation(pend_1m, tf)
    tabFreq = calc_freq_oscillation(pend_1m, tf)
    
    print( tabFreq, np.sqrt(g/L))

    

def aff_traj_pend_2_maillons(pend_2m, N, h, L1, L2):
    """ La fonction "aff_courbe_freq_oscillation" prend 2 arguments :
        - pend_2m : classe représentant le pendule à 2 maillons
        - N : nombre de pas
        - h : pas 
        - L1 : longueur de la tige 1
        - L2 : longueur de la tige 2
        Affiche la trajectoire de l'extrémité du pendule à deux maillons en fonction du temps
    """
    plt.title('Trajectoire pendule 2 maillons')
    plt.xlabel('x')
    plt.ylabel('y')
    
    res = pend_2m.meth_n_step(N, h, pend_2m.step_runge_kutta)
    
    tabX1 = L1* np.sin(res[:,0])
    tabY1 = - L1* np.cos(res[:,0])
    tabX2 = tabX1 + L2 *np.sin(res[:,1])
    tabY2 = tabY1 - L2 *np.cos(res[:,1])
    
    plt.plot(tabX1,tabY1)
    plt.plot(tabX2,tabY2)

    plt.show()
    plt.savefig("trajectoire_pend_2_maillons.png")
    plt.close()



def pendule_deux_maillons():
    L1 = 1.0                # longueur de la tige 1
    m1 = 1.0                # masse du solide 1
    L2 = 0.9                # longueur de la tige 2
    m2 = 0.9                # masse du solide 2
    
    f2 = lambda X, t:  np.array([X[2],
                            X[3],
                            -g*(2*m1+m2)*np.sin(X[0]) - m2 * g * np.sin(X[0]-2*X[1]) - 2 * np.sin(X[0]-X[1]) * m2 * (X[3]*X[3] * L2 + X[2]*X[2] * L1 * np.cos(X[0]-X[1]))/(L1*(2*m1+m2-m2*np.cos(2*X[0]-2*X[1]))),
                            2*np.sin(X[0]-X[1])*(X[2]*X[2] * L1 * (m1 + m2) + g * (m1 + m2) * np.cos(X[0]) + X[3]*X[3] * L2 * m2 * np.cos(X[0] - X[1])) / (L2*(2*m1+m2-m2*np.cos(2*X[0]-2*X[1]))) ])
    pas = 0.01
    nbPas = 3000
    theta0_m1 = np.pi/2
    theta0_m2 = np.pi/2
    
    pend2m = classe.pCauchy(0, [theta0_m1, theta0_m2, 0.0, 0.0], f2)

    aff_traj_pend_2_maillons(pend2m, nbPas, pas, L1, L2)




if __name__ ==  '__main__':
    #pendule_un_maillon()
    pendule_deux_maillons()
    
    
