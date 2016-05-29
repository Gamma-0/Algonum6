# -*- coding: utf-8 -*-

from classe import *


"""
Resolution modele Malthus
"""
def malthus():
    """ 
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
    """ 
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
    """ 
        Simule le modele de lotka-volterra:
           - varCouple.png : graphe des variations des proies et des prédateurs en fonction du temps
           - affiche la période des solutions
           - proie_predateur.png : graphe de l'evolution des predateurs en fonction des proies
           - compLocal.png : graphe montrant le comportement local de plusieurs solutions
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
    
    graph_couple(lv, tf)
    print("La periode des solution est : ", solution_period(lv, tf))
    graph_proie_pred(lv, tf)
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



def solution_period(equDiff, tf):
    """ La fonction "solution_period" prend 2 arguments :
        - equDiff : classe représentant le problème de Cauchy
        - tf : borne max des abscisses
        Retourne la periode des solutions
    """
    def extrem_loc(derivative):
        """ La fonction "extrem_loc" prend 1 argument :
           - derivative : tableau contenant les derivees
           Retourne un tableau contenant les rangs des extremums
        """        
        extrem = []
        last = derivative[0]
        for i in range(1, np.shape(derivative)[0]):
            if( (derivative[i] > 0 and last < 0) or # minimum local
                (derivative[i] < 0 and last > 0) ): # maximum local
                extrem.append(i)
            last = derivative[i]
        return extrem

    sol = equDiff.meth_epsilon(tf, 10e-3, equDiff.step_euler)
    size = np.shape(sol)[0]
    t_sol = np.linspace(0, tf, size, endpoint=False)

    #la periode de variation des proies est la meme que celle des predateurs
    #prenons donc les proies pour le calcul
    sol = sol[:,0]

    #recherche d'extremum locaux
    derivative = [(sol[i+1] - sol[i]) / (t_sol[i+1] - t_sol[i]) for i in range(size-1)]
    extremum = extrem_loc(derivative)
    
    #la periode contient exactement 2 extremums locaux dans ce type de solution
    period = 2 * ((t_sol[max(extremum)] - t_sol[min(extremum)]) / (len(extremum) - 1))
    return period

if __name__ ==  '__main__':
    lotka_volterra()
    malthus()
    verhulst()
