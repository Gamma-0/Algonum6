#http://www.tangentex.com/PenduleMaple.htm

#initialisation des variables
import numpy as np
import math
import matplotlib.pyplot as plt
import classe

g = 9.81                #acceleration de la pesanteur
l = 1.0                 #longueur du pendule
omega0 = 2*np.pi   #pulsation


def equation_pendule(theta0, theta, omega0):
    return math.pow(theta0, 2)+2*math.pow(omega0, 2)*(math.cos(theta0) - math.cos(theta))

def equadiff(X,t):
    theta,theta_point=X
    return np.array([theta_point, -g/l*np.sin(theta)])


theta0 = np.array([np.pi/2, 0])

s = classe.pCauchy(0.0, theta0, equadiff)
res = s.aff_courbe_eq_diff(10, 0.0001)