# -*- coding: utf-8 -*-

from classe import *

t0 = 0.0
y0 = [5.0]
tf = 10


"""
Resolution modele Malthus
"""
def malthus():
    """ La fonction "malthus" prend ... arguments :
        - ... : ...
        Retourne ...
    """
    b = 12.0
    d = 11.0
    
    f_malthus = lambda yt, t: b * yt - d * yt
    
    malthus = pCauchy(t0, y0, f_malthus)
    malthus.aff_courbe_eq_diff(tf, file_name="malthus")


"""
Resolution modele Verhulst
"""
def verhulst():
    """ La fonction "verhulst" prend ... arguments :
        - ... : ...
        Retourne ...
    """
    gamma = 1.5
    kappa = 10e4

    f_verhulst = lambda yt, t: gamma * yt * (1 - yt / kappa)

    verhulst = pCauchy(t0, y0, f_verhulst)
    verhulst.aff_courbe_eq_diff(tf, file_name="verhulst")


"""
Resolution modele Lotka-Volterra
"""
def lotka_volterra():
    """ La fonction "lotka_volterra" prend ... arguments :
        - ... : ...
        Retourne ...
    """
    a = 2.0/3
    b = 4.0/3
    c = 1.0
    d = 1.0
    y0 = np.array([1.0, 1.0])
    tf = 30.0
    
    f_lv = lambda yt, t: np.array([yt[0] * (a - b * yt[1]), yt[1] * (c * yt[0] - d)])
    lv = pCauchy(t0, y0, f_lv)
    #lv.aff_courbe_eq_diff(tf, file_name="lotka-volterra")
    graph_couple(lv)
    graph_proie_pred(lv)



def graph_couple(equDiff):
    """ La fonction "graph_couple" prend ... arguments :
        - ... : ...
        Retourne ...
    """
    sol = equDiff.meth_epsilon(tf, 10e-3, equDiff.step_euler)
    t_sol = np.linspace(0, tf, np.shape(sol)[0], endpoint=False)
    
    plt.title("Variations du couple proies/predateurs")
    plt.xlabel("x")
    plt.ylabel("y")
    
    plt.plot(t_sol, sol[:,0], linewidth=1.0, label="Proies")
    plt.plot(t_sol, sol[:,1], linewidth=1.0, label="Predateurs")

    plt.legend(loc='best')
    plt.show()
    plt.savefig("varCouple.png")
    plt.close()



def graph_proie_pred(equDiff):
    """ La fonction "graph_proie_pred" prend ... arguments :
        - ... : ...
        Retourne ...
    """
    sol = equDiff.meth_epsilon(tf, 10e-3, equDiff.step_euler)

    plt.title("Nombre de predateurs en fonction des proies")
    plt.xlabel("Proies")
    plt.ylabel("Predateurs")
    
    plt.plot(sol[:,0], sol[:,1], linewidth=1.0)

    plt.show()
    plt.savefig("proie_predateur.png")
    plt.close()






if __name__ ==  '__main__':
    lotka_volterra()

