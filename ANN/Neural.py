import numpy as np
import functools
class Generic() : 
    def __init__(self) :
        self.nb_params=None # Nombre de parametres de la couche
        self.save_X=None # Parametre de sauvegarde des donnees
    def set_params(self,params) : 
        # Permet de modifier les parametres de la couche, en entree, prend un vecteur de la taille self.nb_params
        pass
    def get_params(self) : 
        # Rend un vecteur de taille self.params qui contient les parametres de la couche
        return None
    def forward(self,X) : 
        # calcul du forward, X est le vecteur des donnees d'entrees
        self.save_X=np.copy(X)
        return None
    def backward(self,grad_sortie) :  
        # retropropagation du gradient sur la couche, 
        #grad_sortie est le vecteur du gradient en sortie
        #Cette fonction rend :
        #grad_local, un vecteur de taille self.nb_params qui contient le gradient par rapport aux parametres locaux
        #grad_entree, le gradient en entree de la couche 
        grad_local=None
        grad_entree=None
        return grad_local,grad_entree
    
class Arctan() : 
    def __init__(self) :
        self.nb_params=0 # Nombre de parametres de la couche
        self.save_X=None # Parametre de sauvegarde des donnees
    def set_params(self,params) : 
        # Permet de modifier les parametres de la couche, en entree, prend un vecteur de la taille self.nb_params
        pass
    def get_params(self) : 
        # Rend un vecteur de taille self.params qui contient les parametres de la couche
        return None
    def forward(self,X) : 
        # calcul du forward, X est le vecteur des donnees d'entrees
        self.save_X=np.copy(X)
        Y = np.arctan(X)
        return Y
    def backward(self,grad_sortie) :  
        # retropropagation du gradient sur la couche, 
        #grad_sortie est le vecteur du gradient en sortie
        #Cette fonction rend :
        #grad_local, un vecteur de taille self.nb_params qui contient le gradient par rapport aux parametres locaux
        #grad_entree, le gradient en entree de la couche 
        grad_local=np.array([])
        grad_entree= grad_sortie/(1+np.power(self.save_X,2))
        return grad_local,grad_entree
    
class Sigmoid() : 
    def __init__(self) :
        self.nb_params=0 # Nombre de parametres de la couche
        self.save_X=None # Parametre de sauvegarde des donnees
    def set_params(self,params) : 
        # Permet de modifier les parametres de la couche, en entree, prend un vecteur de la taille self.nb_params
        pass
    def get_params(self) : 
        # Rend un vecteur de taille self.params qui contient les parametres de la couche
        return None
    
    def sig(self,X):
        return 1/(1+np.exp(-X))
    
    def forward(self,X) : 
        # calcul du forward, X est le vecteur des donnees d'entrees
        self.save_X=np.copy(X)
        return self.sig(X)
    
    def backward(self,grad_sortie) :  
        # retropropagation du gradient sur la couche, 
        #grad_sortie est le vecteur du gradient en sortie
        #Cette fonction rend :
        #grad_local, un vecteur de taille self.nb_params qui contient le gradient par rapport aux parametres locaux
        #grad_entree, le gradient en entree de la couche 
        grad_local=np.array([])
        grad_entree= grad_sortie * self.sig(self.save_X)*(1-self.sig(self.save_X))
        return grad_local,grad_entree
    
   
class Dense() : 
    def __init__(self, nb_entree, nb_sortie) :
        self.nb_params=(nb_entree+1)*nb_sortie # Nombre de parametres de la couche
        self.save_X=None # Parametre de sauvegarde des donnees
        self.n_entree = nb_entree
        self.n_sortie = nb_sortie
        self.A = np.random.randn(self.n_sortie, self.n_entree)
        self.b = np.random.randn(self.n_sortie)
     
    def set_params(self,params) : 
        # Permet de modifier les parametres de la couche, en entree, prend un vecteur de la taille self.nb_params
        self.A = np.resize(params[:self.n_entree*self.n_sortie], (self.n_sortie, self.n_entree))
        self.b = np.resize(params[self.n_entree*self.n_sortie:],(self.n_sortie))
       
    
    def get_params(self) : 
        # Rend un vecteur de taille self.params qui contient les parametres de la couche
        A = np.ravel(self.A)
        b = np.ravel(self.b)
        return np.append(A,b)
    
    def forward(self,X) : 
        # calcul du forward, X est le vecteur des donnees d'entrees
        self.save_X=np.copy(X)
        Y = self.A@X+np.outer(self.b, np.ones(X.shape[1]))
        return Y
    
    def backward(self,grad_sortie) :  
        # retropropagation du gradient sur la couche, 
        #grad_sortie est le vecteur du gradient en sortie
        #Cette fonction rend :
        #grad_local, un vecteur de taille self.nb_params qui contient le gradient par rapport aux parametres locaux
        #grad_entree, le gradient en entree de la couche 
        ga = grad_sortie @ np.transpose(self.save_X)
        gb = np.sum(grad_sortie, axis=1)
        grad_local=np.append(np.ravel(ga), np.ravel(gb))
        grad_entree= np.transpose(self.A) @ grad_sortie
        return grad_local,grad_entree
    
class Loss_L2() : 
    def __init__(self, D) :
        self.nb_params=0 # Nombre de parametres de la couche
        self.save_X=None # Parametre de sauvegarde des donnees
        self.D = D

    def set_params(self,params) : 
        # Permet de modifier les parametres de la couche, en entree, prend un vecteur de la taille self.nb_params
        pass
    
    def get_params(self) : 
        # Rend un vecteur de taille self.params qui contient les parametres de la couche
        return None
    
    def forward(self,X) : 
        # calcul du forward, X est le vecteur des donnees d'entrees
        self.save_X=np.copy(X)
        Y = np.linalg.norm(X-self.D)**2 /2
        return Y
    
    def backward(self,grad_sortie) :  
        # retropropagation du gradient sur la couche, 
        #grad_sortie est le vecteur du gradient en sortie
        #Cette fonction rend :
        #grad_local, un vecteur de taille self.nb_params qui contient le gradient par rapport aux parametres locaux
        #grad_entree, le gradient en entree de la couche 
        grad_local=np.array([])
        grad_entree= self.save_X - self.D
        return grad_local,grad_entree

class Ilogit_and_KL() : 
    def __init__(self, y) :
        self.nb_params=0 # Nombre de parametres de la couche
        self.save_y_tilde=None # Parametre de sauvegarde des donnees
        self.y = y
        
    def set_params(self,params) : 
        pass
    def get_params(self) : 
        return np.array([])
    
    def ilogit(self,z) :
        x = np.exp(z)
        return x/x.sum(axis=0)
    
    def forward(self,z) : 
        # calcul du forward, X est le vecteur des donnees d'entrees
        self.save_y_tilde=np.copy(self.ilogit(z))
        return np.sum(-np.sum(self.y * np.log(self.save_y_tilde), axis=0))
    
    def backward(self,grad_sortie) :  
        # retropropagation du gradient sur la couche, 
        #grad_sortie est le vecteur du gradient en sortie
        #Cette fonction rend :
        #grad_local, un vecteur de taille self.nb_params qui contient le gradient par rapport aux parametres locaux
        #grad_entree, le gradient en entree de la couche 
        grad_local=np.array([])
        grad_entree= self.save_y_tilde - self.y
        return grad_local,grad_entree
 
    
class Network() : 
    def __init__(self, list_layers) :
 
        self.nb_params = np.sum(np.array([list_layers[i].nb_params for i in range(len(list_layers))]))# Nombre de parametres de la couche
        self.list_layers=list_layers
        self.save_X=None # Parametre de sauvegarde des donnees
    def set_params(self,params) : 
        s = 0
        for i in range(len(self.list_layers)):
            nb_params = self.list_layers[i].nb_params
            if nb_params != 0:
                self.list_layers[i].set_params(params[s:s+nb_params])
                s+= nb_params
    
    def get_params(self) : 
        params = np.array([])
        for i in range(len(self.list_layers)):
            if(self.list_layers[i].get_params() is not None):
                params = np.append(params, self.list_layers[i].get_params())
        return params
    
    def forward(self,X) : 
        # calcul du forward, X est le vecteur des donnees d'entrees
        self.save_X=np.copy(X)
        Z = np.copy(X)
        for i in range(len(self.list_layers)):
            Z = self.list_layers[i].forward(Z)
        return Z
    
    def backward(self,grad_sortie) :  
        # retropropagation du gradient sur la couche, 
        #grad_sortie est le vecteur du gradient en sortie
        #Cette fonction rend :
        #grad_local, un vecteur de taille self.nb_params qui contient le gradient par rapport aux parametres locaux
        #grad_entree, le gradient en entree de la couche 
        grad_local=np.array([])
        grad_entree=grad_sortie
        for L in reversed(self.list_layers):
            a,b = L.backward(grad_entree)
            grad_local = np.insert(grad_local, 0, a, axis=None)
            grad_entree = b
        return grad_local,grad_entree