import numpy as np

def plu_decomposition(A):
    n = A.shape[0]
    P = np.identity(n)
    L = np.identity(n)
    U = A.copy()

    for k in range(n):
        # Recherche du pivot
        max_row = np.argmax(np.abs(U[k:, k])) + k
        if k != max_row:
            # Échanger les lignes dans U
            U[[k, max_row], :] = U[[max_row, k], :]
            # Échanger les lignes dans P
            P[[k, max_row], :] = P[[max_row, k], :]
            # Échanger les lignes dans les premières colonnes de L
            if k > 0:
                L[[k, max_row], :k] = L[[max_row, k], :k]
        
        # Enregistrement des multiplicateurs dans L et mise à jour de U
        for i in range(k+1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] -= L[i, k] * U[k, k:]
    
    # Ajouter la diagonale de 1 à L
    np.fill_diagonal(L, 1)
    
    return P, L, U

# Exemple d'utilisation
A = np.array([[2, 3, 1],
              [4, 7, 2],
              [6, 18, -1]], dtype=float)

P, L, U = plu_decomposition(A)

print("Matrice P:")
print(P)
print("Matrice L:")
print(L)
print("Matrice U:")
print(U)

print("L*U :")
print(L*U)
print("P*A :")
print(P*A)