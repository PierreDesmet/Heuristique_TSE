import pandas as pd
import numpy as np
class Heuristique_TSE():
    """
        @author : b011ldg
        À partir des variables de df (df ne contient QUE des features de X), crée l'interaction X_i*X_j ssi 
        Cor(interaction, cible) > max(Cor(feature_i, cible))

        Pensez à vérifier sur un échantillon de validation que les interactions proposées sont significatives ! 
        
        -------
        Params:
        - df        : ensemble de FEATURES continues, pour lesquelles calculer la corrélation a du sens. /!\ Ne doit inclure ni les données Test, ni la target ; 
        - target    : la Serie cible, p. ex. : df.y_train
        - marge     : l'epsilon supplémentaire que l'on prend pour s'assurer de la significativité de l'intéraction créée, et ne pas créer trop de combinaisons
                      qui "noieraient" les features existantes. Dans ce cas là, on a : Cor(interaction, cible) > max(Cor(feature_i, cible))
        - corr_mode : le mode de calcul de la corrélation : 'pearson', ou 'spearman'
        - interaction_mode : le mode de calcul des interactions : "*" (Xi*Xj), "/" (Xi/Xj), "+" (Xi+Xj), ou "-" (Xi-Xj).
    """
    
    
    def __init__(self, df, target, marge=1e-2, verbose=True, corr_mode='pearson', interaction_mode="*", valid=None):
        self.df = df
        self.target = target
        self.marge = marge
        self.verbose = verbose
        self.corr_mode = corr_mode
        if self.corr_mode == 'pearson' :
            try : 
                assert (abs(df.mean().mean()) < 0.1) and ((df.std().mean() > 0.9) and (df.std().mean() < 1.1))
            except : raise NameError("Les données doivent être normalisées pour la corrélation peason !") 
        self.interaction_mode = interaction_mode
        self.interactions_à_créer = []
        self.corr_min = None
        self.corr_max = None
        
        
    def make_interactions(self):
        from scipy.stats import spearmanr
        # 1 - Calcul des corrélations (X_i, cible) :
        if self.corr_mode == 'pearson'  : correlations = [np.corrcoef(self.df[var], self.target)[0,1] for var in self.df]
        if self.corr_mode == 'spearman' : correlations = [spearmanr(self.df[var], self.target).correlation for var in self.df] 
        self.corr_min, self.corr_max = min(correlations), max(correlations)
        print(f"La corrélation ({self.corr_mode}) minimale avec la cible est {self.corr_min:.4f} ; la corrélation maximale avec la cible est {self.corr_max:.4f}.")

        # 2 - Création des interactions X_i * X_j. Seulement celles qui méritent de l'être :
        from itertools import combinations
        nb_combinaisons, itération = len(list(combinations(self.df.columns, 2))), 0
        print('Itération', end=' ')
        
        def opération(col1, col2, opérateur="*"):
            if opérateur=="*" : return col1*col2
            if opérateur=="-" : return col1-col2
            if opérateur=="+" : return col1+col2
            if opérateur=="/" : return col1/col2
            
        for var_i, var_j in combinations(self.df.columns, 2):
            
            if self.corr_mode=='pearson'  : c = np.corrcoef(opération(self.df[var_i], self.df[var_j], self.interaction_mode), self.target)[0,1]
            if self.corr_mode=='spearman' : c = spearmanr(opération(self.df[var_i], self.df[var_j], self.interaction_mode), self.target).correlation
            
            if (c < self.corr_min-self.marge) or (c > self.corr_max+self.marge) : self.interactions_à_créer.append((var_i, var_j, np.round(c, 4)))

            if self.verbose and (itération%(nb_combinaisons//10) == 0) : print(f"{itération}/{nb_combinaisons} ;", end='  ')
            itération += 1


    def valide_interaction(self, X_valid, y_valid, remove=False):
        """
        Valide que les interactions suggérées par l'heuristique du train fonctionnent également sur un échantillon de validation, lequel n'a pas été utilisé pour les calculer.
        """
        self.X_valid = X_valid
        self.y_valid = y_valid
        for tup in self.interactions_à_créer:
            c = np.corrcoef(self.X_valid[tup[0]] * self.X_valid[tup[1]],  self.y_valid)[0,1]
            if (c > self.corr_min-self.marge) and (c < self.corr_max+self.marge) :
                print(f"///Corr. entre {tup[0]} et {tup[1]} = {c:.4f}, significative sur le train mais pas sur le valid !///")
                if remove : self.interactions_à_créer.remove(tup)
    
    def change_marge(self, new_marge):
        """
        Change la self.marge actuelle, et retire en conséquence les self.interactions_à_créer qui n'ont pas une corrélation suffisamment élevée avec la cible.
        """
        self.marge = new_marge
        self.interactions_à_créer = [it for it in self.interactions_à_créer if (it[2]>self.corr_max+self.marge) or (it[2]<self.corr_min-self.marge)]