# -*- coding: utf-8 -*-

import numpy as np
import numpy.linalg as lg

# --- Méthodes numériques de résolution d'équations différentielles --- #

# --- Méthode d'Euler --- #

def euler(y, t, h, f):

    res = y
    k = 0

    for i in range(0, y.shape[0]):
        res[i] = y[i] + h[i] * f(t[i], y[i])
        
    return res


# --- Méthode du point milieu --- #

def point_milieu(y, t, h, f):

    
