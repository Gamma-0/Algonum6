#http://www.tangentex.com/PenduleMaple.htm

#initialisation des variables
import numpy as np
import math
import classe

g = 9.81                #acceleration de la pesanteur
l = 1.0                 #longueur du pendule
omega0 = np.sqrt(g/l)   #pulsation


def equation_pendule(theta0, theta, omega0):
    return math.pow(theta0, 2)+2*math.pow(omega0, 2)*(math.cos(theta0) - math.cos(theta))

#(application du principe fondamental de la dynamique)
# on a deriv_theta = -g/l * sin(theta), mais comme on fait de petites oscillations, sin(theta) = theta
def deriv_theta(theta):
    return -(g/l)*theta

T0 = 2*math.pi*(math.sqrt(g/l))

abscissas = [i for i in range(0, 1, 0.01)]

s = classe.pCauchy()