from classe import *

t0 = 0.0

y0 = [5.0]

tf = 10


"""

Resolution modele Malthus



b = 12.0

d = 11.0

def f_malthus(yt, t):
    return b * yt - d * yt

malthus = pCauchy(t0, y0, f_malthus)

malthus.aff_courbe_eq_diff(tf, file_name="malthus")

"""

"""

Resolution modele Verhulst



gamma = 1.5

kappa = 10e4

def f_verhulst(yt, t):
    return gamma * yt * (1 - yt / kappa)

verhulst = pCauchy(t0, y0, f_verhulst)

verhulst.aff_courbe_eq_diff(tf, file_name="verhulst")

"""

"""

Resolution modele Lotka-Volterra

"""

a = 2.0/3

b = 4.0/3

c = 1.0

d = 1.0

y0 = np.array([1.0, 1.0])

tf = 30.0

def f_lv(yt, t):
    return np.array([yt[0] * (a - b * yt[1]), yt[1] * (c * yt[0] - d)])

lv = pCauchy(t0, y0, f_lv)
#lv.aff_courbe_eq_diff(tf, file_name="lotka-volterra")

def graph_couple():
    sol = lv.meth_epsilon(tf, 10e-3, lv.step_euler)
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


def graph_proie_pred():
    sol = lv.meth_epsilon(tf, 10e-3, lv.step_euler)

    plt.title("Nombre de predateurs en fonction des proies")
    plt.xlabel("Proies")
    plt.ylabel("Predateurs")
    
    plt.plot(sol[:,0], sol[:,1], linewidth=1.0)

    plt.show()
    plt.savefig("proie_predateur.png")
    plt.close()


graph_couple()

