import numpy as np
import matplotlib.pyplot as plt

def simulation_MM1(lmbda, mu, n_sim):
    """
    Simulation d'une file d'attente M/M/1
    lmbda: taux d'arrivee (λ)
    mu: taux de service (µ)
    n_sim: nombre de clients à simuler
    """
    # Initialisation
    N = []      # nombre total de clients
    Nq = []     # nombre de clients en attente
    B = []      # état du serveur (0=inactif, 1=actif)
    
    # Premier client
    tin = np.random.exponential(1/lmbda)
    Ws = np.random.exponential(1/mu)
    
    Tq = 0        # délai d'attente
    t = tin       # temps actuel
    N.append(1)
    Nq.append(0)
    B.append(1)
    
    last_departure = t + Ws  # temps de fin de service du premier client
    
    # Simulation des autres clients
    for i in range(1, n_sim):
        tin = np.random.exponential(1/lmbda)
        t += tin  # temps d'arrivée du client i
        
        Ws = np.random.exponential(1/mu)
        
        # Calcul du délai d'attente
        Tq = max(0, last_departure - t)
        
        # Mise à jour du serveur et du nombre de clients
        if last_departure <= t:
            # serveur inactif, aucun client en attente
            B.append(0)
            Nq.append(0)
            N.append(1)  # seul le client qui arrive
        else:
            # serveur actif
            B.append(1)
            Nq.append(max(0, N[-1]))  # clients en attente
            N.append(N[-1] + 1)
        
        last_departure = max(last_departure, t) + Ws  # mise à jour du temps de départ
    
    # Conversion en arrays pour facilité
    N = np.array(N)
    Nq = np.array(Nq)
    B = np.array(B)
    
    return N, Nq, B

# Paramètres
lmbda = 0.8  # taux d'arrivée
mu = 1.0     # taux de service
n_sim = 100  # nombre de clients simulés

# Simulation
N, Nq, B = simulation_MM1(lmbda, mu, n_sim)

# Graphiques
plt.figure(figsize=(12,5))
plt.plot(N, label="N (clients totaux)")
plt.plot(Nq, label="Nq (clients en attente)")
plt.plot(B, label="B (serveur actif=1/inactif=0)")
plt.xlabel("Client index")
plt.ylabel("Nombre / état")
plt.title("Simulation M/M/1")
plt.legend()
plt.grid(True)
plt.show()

# Calculs statistiques
L_sim = np.mean(N)
Lq_sim = np.mean(Nq)
pourcentage_inoccupation = np.mean(1-B) * 100

print(f"Nombre moyen de clients dans la file (L) = {L_sim:.2f}")
print(f"Nombre moyen de clients en attente (Lq) = {Lq_sim:.2f}")
print(f"Pourcentage d'inoccupation du serveur = {pourcentage_inoccupation:.2f}%")

rho = lmbda / mu
L_theorique = rho / (1 - rho)
Lq_theorique = rho**2 / (1 - rho)
U_theorique = rho * 100  

print("\nComparaison avec la théorie:")
print(f"L théorique = {L_theorique:.2f}, Lq théorique = {Lq_theorique:.2f}, %occupation serveur = {U_theorique:.2f}%")