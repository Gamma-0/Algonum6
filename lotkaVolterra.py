# -*- coding: utf-8 -*-

from classe import *


"""
Resolution modele Malthus
"""
def malthus():
    """ La fonction "malthus" prend ... arguments :
        - ... : ...
        Affiche la courbe de la variation de la population au cours du temps, en suivant le modèle de Malthus
    """
    t0 = 0.0
    y0 = [5.0]
    tf = 10
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
        Affiche la courbe de la variation de la population au cours du temps, en suivant le modèle de Verhulst
    """
    t0 = 0.0
    y0 = [5.0]
    tf = 10
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
    a = 2.0/3       # taux de reproduction des proies
    b = 4.0/3       # taux de mortalité des proies
    c = 1.0         # taux de reproduction des prédateurs
    d = 1.0         # taux de mortalité des prédateurs
    t0 = 0.0
    y0 = np.array([1.0, 1.0])
    tf = 30.0
    
    f_lv = lambda yt, t: np.array([yt[0] * (a - b * yt[1]), yt[1] * (c * yt[0] - d)])
    lv = pCauchy(t0, y0, f_lv)
    #graph_couple(lv, tf)
    #graph_proie_pred(lv, tf)
    graph_comportement_local(lv, 5.0)



def graph_couple(equDiff, tf):
    """ La fonction "graph_couple" prend 2 arguments :
        - equDiff : classe représentant le problème de Cauchy
        - tf : borne max des abscisses
        Affiche la courbe des variations du couple (Proies, Prédateurs) au cours du temps.
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



def graph_proie_pred(equDiff, tf):
    """ La fonction "graph_proie_pred" prend 2 arguments :
        - equDiff : classe représentant le problème de Cauchy
        - tf : borne max des abscisses
        Affiche les interactions de deux populations proies/prédateurs
    """
    sol = equDiff.meth_epsilon(tf, 10e-3, equDiff.step_euler)

    plt.title("Nombre de predateurs en fonction des proies")
    plt.xlabel("Proies")
    plt.ylabel("Predateurs")
    
    plt.plot(sol[:,0], sol[:,1], linewidth=1.0)

    plt.show()
    plt.savefig("proie_predateur.png")
    plt.close()



def local(y0, eps=0.2):
    """ La fonction "local" prend 2 arguments :
        - y0 : point d'origine (en dimension 2)
        - eps : variations sur la coordonnée 2
        Retourne un tableau contenant les points voisins de y0
    """
    loc = []
    for j in np.arange(y0[1] - eps, y0[1] + eps, eps / 10):
        loc.append([y0[0], j])
    return loc



def graph_comportement_local(equDiff, tf):
    """ La fonction "graph_comportement_local" prend 2 arguments :
        - equDiff : classe représentant le problème de Cauchy
        - tf : borne max des abscisses
        Affiche le graphe du comportement local de l'équation différentielle.
    """
    plt.title("Comportement local des solutions")
    plt.xlabel("Proies")
    plt.ylabel("Predateurs")
    plt.axis([0.45,0.65,0.6,0.75])
    y0 = local(equDiff.y0)

    for equDiff.y0 in y0:
        sol = equDiff.meth_epsilon(tf, 10e-3, equDiff.step_euler)
        plt.plot(sol[:,0], sol[:,1], linewidth=1.0, color='k')
    plt.show()
    plt.savefig("compLocal.png")
    plt.close()



if __name__ ==  '__main__':
    lotka_volterra()
    #malthus()
    #verhulst()

