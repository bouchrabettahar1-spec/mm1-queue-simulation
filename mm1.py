import numpy as np
import matplotlib.pyplot as plt

def simulation_MM1(lmbda, mu, n_sim):
    """
    Simulation d'une file d'attente M/M/1
    lmbda: taux d'arrivee (λ)
    mu: taux de service (µ)
    n_sim: nombre de clients à simuler
    """
    N = []      
    Nq = []    
    B = []     
    
   
    tin = np.random.exponential(1/lmbda)
    Ws = np.random.exponential(1/mu)
    
    Tq = 0       
    t = tin       
    N.append(1)
    Nq.append(0)
    B.append(1)
    
    last_departure = t + Ws  
    for i in range(1, n_sim):
        tin = np.random.exponential(1/lmbda)
        t += tin 
        
        Ws = np.random.exponential(1/mu)
      
        Tq = max(0, last_departure - t)
        
       
        if last_departure <= t:
            
            B.append(0)
            Nq.append(0)
            N.append(1)  
        else:
           
            B.append(1)
            Nq.append(max(0, N[-1]))  
            N.append(N[-1] + 1)
        
        last_departure = max(last_departure, t) + Ws  
    
    N = np.array(N)
    Nq = np.array(Nq)
    B = np.array(B)
    
    return N, Nq, B


lmbda = 0.8  
mu = 1.0     
n_sim = 100  


N, Nq, B = simulation_MM1(lmbda, mu, n_sim)


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