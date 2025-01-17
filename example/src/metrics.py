from skimage.measure import label
from sklearn.metrics import confusion_matrix
import numpy as np

def Intersection(A, B):
    """
    Returns the pixel count corresponding to the intersection
    between A and B.
    """
    C = A + B
    C[C != 2] = 0
    C[C == 2] = 1
    return C

def Union(A, B):
    """
    Returns the pixel count corresponding to the union
    between A and B.
    """
    C = A + B
    C[C > 0] = 1
    return C


def AssociatedCell(G_i, S):
    """
    Returns the indice of the associated prediction cell for a certain
    ground truth element. Maybe do something if no associated cell in the 
    prediction mask touches the GT.
    """
    def g(indice):
        S_indice = np.zeros_like(S)
        S_indice[ S == indice ] = 1
        NUM = float(Intersection(G_i, S_indice).sum())
        DEN = float(Union(G_i, S_indice).sum())
        return NUM / DEN
    res = map(g, range(1, S.max() + 1))
    indice = np.array(res).argmax() + 1
    return indice

def AJI(G, S):
    """
    AJI as described in the paper, but a much faster implementation.
    """
    G = label(G, background=0)
    S = label(S, background=0)
    if S.sum() == 0:
        return 0.
    C = 0
    U = 0 
    USED = np.zeros(S.max())

    G_flat = G.flatten()
    S_flat = S.flatten()
    G_max = np.max(G_flat)
    S_max = np.max(S_flat)
    m_labels = max(G_max, S_max) + 1
    cm = confusion_matrix(G_flat, S_flat, labels=range(m_labels)).astype(np.float)
    LIGNE_J = np.zeros(S_max)
    for j in range(1, S_max + 1):
        LIGNE_J[j - 1] = cm[:, j].sum()

    for i in range(1, G_max + 1):
        LIGNE_I_sum = cm[i, :].sum()
        def h(indice):
            LIGNE_J_sum = LIGNE_J[indice - 1]
            inter = cm[i, indice]

            union = LIGNE_I_sum + LIGNE_J_sum - inter
            return inter / union
        
        JI_ligne = map(h, range(1, S_max + 1))
        best_indice = np.argmax(JI_ligne) + 1
        C += cm[i, best_indice]
        U += LIGNE_J[best_indice - 1] + LIGNE_I_sum - cm[i, best_indice]
        USED[best_indice - 1] = 1

    U_sum = ((1 - USED) * LIGNE_J).sum()
    U += U_sum
    return float(C) / float(U)  

