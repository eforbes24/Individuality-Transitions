#### AUTOMATA MODEL ####
## Eden Forbes
## Object Class Definitions

#### LIBRARIES
import numpy as np
import random

#### OBJECTS

class tile:
    CA = False
    CAfood = False
    INDfood = False
    CAgrad = 0.0
    CAFgrad = 0.0
    INDFgrad = 0.0

class world:
    def __init__(self, X, Y, start_CA, CAfood, INDfood):
        CAlist = list()
        CAfoodlist = list()
        INDfoodlist = list()
        
    def generateWorld(X, Y, start_CA, CAfood, INDfood):
        dim = np.asarray([[0] * X for _ in range(Y)])
        dim = dim.shape
        world = list()
        for i in range(X*Y):
            world.append(tile())
            CAinds = np.random.choice(len(world), size=start_CA)
            for i in CAinds:
                tile = world[i]
                tile.CA = True
                CAFinds = np.random.choice(len(world), size=CAfood)
                for i in CAFinds:
                    tile = world[i]
                    tile.CAfood = True
                    INDFinds = np.random.choice(len(world), size=INDfood)
                    for i in INDFinds:
                        tile = world[i]
                        tile.INDfood = True   
                        world = np.asarray(world)
                        world = world.reshape(dim)
        return world
        
    def getFeat(self, feat):
        for i in range(len(self)):
            for j in range(len(self)):
                posx = i
                posy = j
                pos = self[i,j]
                if pos.feat == True:
                    featlist.append((posx,posy))
                    return featlist



class CA:
    def __init__(self,X,Y):
         ## CA Properties
         genome = np.zeros(5)
         ind = None
         repro = True
         diff = False
         fill = random.uniform(0.5,0.7)
        
    def __str__(self):
        return 'Fill: %s. Ind: %s' % (self.fill, self.ind)
   





