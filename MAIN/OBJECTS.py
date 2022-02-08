#### INDIVIDUALITY MODEL ####
## Eden Forbes
## Object Class Definitions

#### LIBRARIES
import numpy as np
import random
from sklearn.datasets import make_blobs
from matplotlib import pyplot
from matplotlib.pyplot import figure

#### FROZEN VARIABLES: DO NOT CHANGE FOR SCIENTIFIC PURPOSES

## Gradient Features
g1 = 0.5 ## Chemotaxis gradient one Manhatten distance from food source
g2 = 0.25 ## Chemotaxis gradient two Manhatten distance from food source
g3 = 0.125 ## Chemotaxis gradient three Manhatten distance from food source
g4 = 0.08 ## Chemotaxis gradient four Manhatten distance from food source
g5 = 0.05 ## Chemotaxis gradient five Manhatten distance from food source
g0 = 20*g1 ## Chemotaxis gradient at food source

## Food Clustering
clusters = 5 ## Number of food clusters in the world
cdev = 10 ## How internally spread clusters are, smaller numbers are greater distances
newcenterstep = 10 ## Number of times during the simulation you want new food clusters to form

## Food Features
enz = 3 ## The amount of food toughness removed per step by a unicellular CA
foodtough = enz ## Minimum max food toughness in simulation
foodregenz = enz/6 ## How much food toughness is regenerated per step
nutrition = foodtough * 0.1 ## ratio of reward to toughness of food
foodloss = 0.01 ## How much food agents lose per step

#####################################
## BASE FUNCTIONS
#####################################

def sigmoid(x):
    return 1/(1+np.exp(-x))

####################################
## DATA FUNCTIONS
####################################
    
def frameconvert(seed,xmin,xmax,ymin,ymax):
    array = seed
    array = array[xmin:xmax]
    newarray = []
    for i in range(len(array)):
        row = array[i]
        newarray.append(row[ymin:ymax])
    return(newarray)

########################################
########################################
##  TILES
########################################                        
########################################

class Tile:
    CA = False
    CAfood = False
    CAgrad = 0.0
    CAFgrad = 0.0
    foodtough = 0.0
    ftcap = 0.0
    foodvalue = 0.0
    x = 0
    y = 0
    desire = 4
    fill = 0

########################################
########################################
##  WORLD
########################################                        
########################################
    
class World:
    def __init__(self, ID, timesteps, X, Y, start_CA, CAfood, 
                 newfood, 
                 foodchance,caf_scal, adhesion_scal, 
                 agent_chemo_scal, display):
        ## Give world attributes
        self.ID = ID
        self.timesteps = timesteps
        self.X = X
        self.Y = Y
        self.foodloss = foodloss
        self.newfood = newfood
        self.foodchance = foodchance
        self.steps = 0
        self.CAs = list()
        ## Gradient & Enzyme attributes
        self.G0 = g0
        self.G1 = g1
        self.G2 = g2
        self.G3 = g3
        self.G4 = g4
        self.G5 = g5
        self.foodtough = foodtough
        self.foodregenz = foodregenz
        self.nutrition = nutrition
        self.caf_scal = caf_scal
        self.adhesion_scal = adhesion_scal
        self.agent_chemo_scal = agent_chemo_scal
        ## Define world dimensions
        dim = np.asarray([[0] * X for _ in range(Y)])
        dim = dim.shape
        self.dim = dim
        self.length = X*Y
        ## Populate world with tiles
        self.tiles = []
        for i in range(self.length):
            space = Tile()
            space.x = i // X
            space.y = i % X
            self.tiles.append(space)
        self.tiles = np.asarray(self.tiles)
        self.tiles = self.tiles.reshape(self.dim)
        ### Generate Food & Cluster
        self.CAfood = CAfood
        self.clusters = clusters
        self.cdev = cdev
        self.newcenterstep = self.timesteps/newcenterstep
        self.centers = np.random.choice(self.X, size=(1,self.clusters))
        self.center_box = (0, self.X) # defines the box that cluster centres are allowed to be in
        self.center_dev = self.X/self.cdev # defines the standard deviation of clusters
        CAFindsX, y = make_blobs(n_samples=self.CAfood, n_features=2, centers=self.centers, center_box=self.center_box, cluster_std=self.center_dev)
        CAFindsX = np.ndarray.tolist(CAFindsX)
        flatX = []
        for i in CAFindsX:
            for j in i:
                flatX.append(j)
        flatX = [round(elem) for elem in flatX]
        flatX = [self.foodwrapper(elem,0) for elem in flatX]    
        CAFindsY, y = make_blobs(n_samples=self.CAfood, n_features=2, centers=self.centers, center_box=self.center_box, cluster_std=self.center_dev)
        CAFindsY = np.ndarray.tolist(CAFindsY)
        flatY = []
        for i in CAFindsY:
            for j in i:
                flatY.append(j)
        flatY = [round(elem) for elem in flatY]
        flatY = [self.foodwrapper(elem,0) for elem in flatY]
        for i in range(len(CAFindsX)):
            til = self.tiles[flatX[i],flatY[i]]
            til.CAfood = True
            til.foodtough = self.foodtough
            til.ftcap = til.foodtough
        ## Generate CAs
        self.enz = enz
        for i in range(start_CA):
            cell = CA(self)
            self.CAs.append(cell)
        ## World Output Data
        self.arrays = list()
        self.data = list()
        self.run = list()
        self.fills = list()
        self.display = display
    
    def __str__(self):
        return 'CA List: %s' % (self.CAs)
    
    def foodwrapper(self, value, dim):
        if dim == 0:
            xlim = self.X - 1
            if value > xlim:
                value = value - self.X
            elif value < 0:
                value = value + self.X
            else:
                value = value
        else:
            ylim = self.Y - 1
            if value > ylim:
                value = value - self.Y
            elif value < 0:
                value = value + self.Y
            else:
                value = value
        return(value)
    
    def wrapper(self, coordlist):
        xlim = self.X - 1
        ylim = self.Y - 1
        finallist = []
        for i in range(len(coordlist)):
            x = coordlist[i][0]
            y = coordlist[i][1]
            if x > xlim:
                x = x - self.X
            else:
                x = x
            if x < 0:
                x = x + self.X
            else:
                x = x
            if y > ylim:
                y = y - self.Y
            else:
                y = y
            if y < 0:
                y = y + self.Y
            else:
                y = y
            finallist.append((x,y))
        return finallist
    
    def showWorld(self):
        array = np.zeros((self.X, self.Y))
        tilearray = self.tiles.flatten()
        for til in tilearray:
            locx = til.x
            locy = til.y
            if til.CA == True:
                ## WHITE
                array[locx,locy] = self.G0 + 9
            elif til.CAfood == True:
                ## TEAL + BLUES
                array[locx,locy] = self.G0 + 1
            else:
                array[locx,locy] = til.CAFgrad
        ## Save array frame
        self.arrays.append(array)
        ## Display the array
        if self.display == True:
            figure(figsize=(8, 8), dpi=70)
            pyplot.imshow(array, interpolation='nearest', cmap='gist_earth',
                          vmin = 0, vmax = self.G0 + 9)
            pyplot.show()
    
    ####################################
    ##  CHEMO GRADIENT FUNCTIONS
    ####################################
    
    def firstDegree(self, x, y):
        coordlist = [(x+1,y), (x,y+1), (x-1,y), (x,y-1)]
        coordlist = self.wrapper(coordlist)
        fList = []
        for i in coordlist:
            fList.append(self.tiles[i[0]][i[1]])
        return fList
    
    def secondDegree(self, x, y):
        coordlist = [(x+1,y+1), (x-1,y+1), (x-1,y-1), (x+1,y-1),
                     (x+2,y), (x,y+2), (x-2,y), (x,y-2)]
        coordlist = self.wrapper(coordlist)
        sList = []
        for i in coordlist:
            sList.append(self.tiles[i[0]][i[1]])
        return sList
    
    def thirdDegree(self, x, y):
        coordlist = [(x+2,y+1), (x-2,y+1), (x-2,y-1), (x+2,y-1),
                     (x+1,y+2), (x-1,y+2), (x-1,y-2), (x+1,y-2),
                     (x+3,y), (x,y+3), (x-3,y), (x,y-3)]
        coordlist = self.wrapper(coordlist)
        tList = []
        for i in coordlist:
            tList.append(self.tiles[i[0]][i[1]])
        return tList
    
    def fourthDegree(self, x, y):
        coordlist = [(x+3,y+1), (x-3,y+1), (x-3,y-1), (x+3,y-1),
                     (x+1,y+3), (x-1,y+3), (x-1,y-3), (x+1,y-3),
                     (x+2,y+2), (x-2,y+2), (x-2,y-2), (x+2,y-2),
                     (x+4,y), (x,y+4), (x-4,y), (x,y-4)]
        coordlist = self.wrapper(coordlist)
        fList = []
        for i in coordlist:
            fList.append(self.tiles[i[0]][i[1]])
        return fList
    
    def fifthDegree(self, x, y):
        coordlist = [(x+3,y+2), (x-3,y+2), (x-3,y-2), (x+3,y-2),
                     (x+2,y+3), (x-2,y+3), (x-2,y-3), (x+2,y-3),
                     (x+4,y+1), (x-4,y+1), (x-1,y-4), (x+1,y-4),
                     (x+1,y+4), (x-1,y+4), (x-4,y-1), (x+4,y-1),
                     (x+5,y), (x,y+5), (x-5,y), (x,y-5)]
        coordlist = self.wrapper(coordlist)
        fiList = []
        for i in coordlist:
            fiList.append(self.tiles[i[0]][i[1]])
        return fiList
    
    def newCenters(self):
        if self.timesteps / self.newcenterstep == self.timesteps // self.newcenterstep:
            self.centers = np.random.choice(self.X, size=(1,self.clusters))
            print("NEW CENTERS")
        
    def addFood(self):
        count = int(self.newfood)
        CAFindsX, y = make_blobs(n_samples=count, n_features=2, centers=self.centers, center_box=self.center_box, cluster_std=self.center_dev)
        CAFindsX = np.ndarray.tolist(CAFindsX)
        flatX = []
        for i in CAFindsX:
            for j in i:
                flatX.append(j)
        flatX = [round(elem) for elem in flatX]
        flatX = [self.foodwrapper(elem,0) for elem in flatX]    
        CAFindsY, y = make_blobs(n_samples=count, n_features=2, centers=self.centers, center_box=self.center_box, cluster_std=self.center_dev)
        CAFindsY = np.ndarray.tolist(CAFindsY)
        flatY = []
        for i in CAFindsY:
            for j in i:
                flatY.append(j)
        flatY = [round(elem) for elem in flatY]
        flatY = [self.foodwrapper(elem,0) for elem in flatY]
        for i in range(len(flatX)):
            til = self.tiles[flatX[i],flatY[i]]
            if til.CA == False:
                if til.CAfood == False:
                    chance = random.uniform(0,1)
                    if chance <= self.foodchance:
                        til.CAfood = True
                        til.foodtough = self.foodtough
                        til.foodvalue = til.foodtough * self.nutrition
                        til.ftcap = til.foodtough
        self.updateCAFGrad()
        self.updateCAFTough()
        
    ####################################
    ##  UPDATE TILE FUNCTIONS
    ####################################
    
    def updateCAGrad(self):
        for row in self.tiles:
            for tile in row:
                tile.CAgrad = 0
        CAx = []
        CAy = []
        for cell in self.CAs:
            CAx.append(cell.x)
            CAy.append(cell.y)
        for i in range(len(CAx)):
            posx = CAx[i]
            posy = CAy[i]
            ## Identity
            self.tiles[posx][posy].CAgrad = self.G0
            ## First Degree
            fList = self.firstDegree(posx,posy)
            for tile in fList:
                tile.CAgrad = tile.CAgrad + self.G1
            ## Second Degree
            sList = self.secondDegree(posx,posy)
            for tile in sList:
                tile.CAgrad = tile.CAgrad + self.G2
            ## Third Degree
            tList = self.thirdDegree(posx,posy)
            for tile in tList:
                tile.CAgrad = tile.CAgrad + self.G3
            fList = self.fourthDegree(posx,posy)
            for tile in fList:
                tile.CAgrad = tile.CAgrad + self.G4
            fiList = self.fifthDegree(posx,posy)
            for tile in fiList:
                tile.CAgrad = tile.CAgrad + self.G5
                
    def updateCAFGrad(self):
        for row in self.tiles:
            for tile in row:
                tile.CAFgrad = 0
                tile.desire = 4
        for i in range(self.X):
            for j in range(self.Y):
                tile = self.tiles[i][j]
                posx = tile.x
                posy = tile.y
                if tile.CAfood == True:
                    ## Identity
                    self.tiles[posx][posy].CAFgrad = self.G0
                    ## First Degree
                    fList = self.firstDegree(posx,posy)
                    for tile in fList:
                        tile.CAFgrad = tile.CAFgrad + self.G1
                    ## Second Degree
                    sList = self.secondDegree(posx,posy)
                    for tile in sList:
                        tile.CAFgrad = tile.CAFgrad + self.G2
                    ## Third Degree
                    tList = self.thirdDegree(posx,posy)
                    for tile in tList:
                        tile.CAFgrad = tile.CAFgrad + self.G3
                    ## Fourth Degree
                    fList = self.fourthDegree(posx,posy)
                    for tile in fList:
                        tile.CAFgrad = tile.CAFgrad + self.G4
                    ## Fifth Degree
                    fiList = self.fifthDegree(posx,posy)
                    for tile in fiList:
                        tile.CAFgrad = tile.CAFgrad + self.G5
                else:
                    tile.CAfood = False
                
    def updateCA(self):
        for row in self.tiles:
            for tile in row:
                tile.CA = False
        for cell in self.CAs:
            posx = cell.x
            posy = cell.y
            self.tiles[posx][posy].CA = True
            
    def updateCAFTough(self):
        foodcount = 0
        for row in self.tiles:
            for tile in row:
                if tile.CAfood == True:
                    foodcount = foodcount + 1
                    tile.foodtough = tile.foodtough + self.foodregenz
                    if tile.foodtough > tile.ftcap:
                        tile.foodtough = tile.ftcap
                    else: 
                        tile.foodtough = tile.foodtough
                else:
                    tile.CAfood = False
        self.CAfood = foodcount
                    
                    
    ####################################
    ##  CHECK CONDITION FUNCTIONS
    ####################################

    def Reproduce(self, cell):
        neighbors = cell.neighbors
        check = 0
        while check == 0:
            if len(neighbors) == 0:
                check = 1
            else:
                loc = random.choice(neighbors)
                if loc.CA == True or loc.CAfood == True:
                    neighbors.remove(loc)
                else:
                    cell.children = cell.children + 1
                    cell.fill = cell.fill/2
                    self.tiles[loc.x][loc.y].CA = True
                    newcell = CA(self)
                    newcell.x = loc.x
                    newcell.y = loc.y
                    newcell.getNeighbors(self)
                    newcell.fill = cell.fill
                    self.CAs.append(newcell)
                    check = 1
                    
    def checkRepro(self):
        for cell in self.CAs:
            cell.getNeighbors(self)
            if cell.fill >= 1:
                self.Reproduce(cell)
                #print("A cell reproduced!")
            else:
                self.CAs = self.CAs
        self.updateCA()
    
    def checkDeath(self):
        for cell in self.CAs:
            if cell.fill <= 0:
                cell.end = self.steps
                self.data.append(["cell", "died", cell.ID, "gen", cell.start, cell.end, 
                                  cell.lifespan, cell.children])
                self.tiles[cell.x][cell.y].CA = False
                self.CAs.remove(cell)
                #print("A cell has died :(")
            else:
                cell.lifespan = cell.lifespan + 1
        self.updateCA()
                
    def checkEnd(self):
        if len(self.CAs) == 0:
            print("All beings have died.")
            return True
        elif self.steps == (self.timesteps - 1):
            self.step()
            print("End Simulation")
            return True
        else:
            return False
        
    ####################################
    ##  STEP FUNCTION
    ####################################
        
    def step(self):
        self.showWorld()
        self.addFood()
        
        ### EAT & ADD FOOD
        for cell in self.CAs:
            ## Each cell eats and update chemical gradients (in function)
            cell.eatWorld(self)
        
        ### MOVE
        self.CAs.sort()
        for cell in self.CAs:
            cell.move_desires(self)
            cell.move_adhesion(self)
            ## Move each cell and update chemical gradiants (in function)
            ## Note that for CAS there are two move functions to choose from,
            ## comment out one.
            ## cell.movebrain(self)
            cell.movesimple(self)
        
        ### CHECKS
        ## Check for cell death and reproduction
        self.checkDeath()
        self.checkRepro()
        
        ### STEP
        nUni = len(self.CAs)
        self.run.append([self.steps,nUni,self.CAfood])
        fills = []
        xs = []
        ys = []
        for cell in self.CAs:
            fills.append(cell.fill)
            xs.append(cell.x)
            ys.append(cell.y)
        self.fills.append([fills,xs,ys])
        self.steps = self.steps + 1
    

########################################
########################################
##  ZOO
########################################                        
########################################
    
class Zoo:
    def __init__(self, ID, seed, seedCAs, timesteps, xmax, xmin, ymax, ymin,
                 newfood,
                 foodchance,caf_scal, adhesion_scal, 
                 agent_chemo_scal,display):
        ## Give world attributes
        self.ID = ID
        self.timesteps = timesteps
        self.seed = seed
        self.seedCAs = seedCAs
        self.xmax = xmax
        self.xmin = xmin
        self.ymax = ymax
        self.ymin = ymin
        self.X = self.xmax-self.xmin
        self.Y = self.ymax-self.ymin
        self.newfood = newfood
        self.foodchance = foodchance
        self.foodloss = foodloss
        self.steps = 0
        self.CAs = list()
        ## Gradient & Enzyme attributes
        self.G0 = g0
        self.G1 = g1
        self.G2 = g2
        self.G3 = g3
        self.G4 = g4
        self.G5 = g5
        self.foodtough = foodtough
        self.foodregenz = foodregenz
        self.nutrition = nutrition
        self.caf_scal = caf_scal
        self.adhesion_scal = adhesion_scal
        self.agent_chemo_scal = agent_chemo_scal
        ## Define world dimensions
        dim = np.asarray([[0] * self.X for _ in range(self.Y)])
        dim = dim.shape
        self.dim = dim
        self.length = self.X*self.Y
        ## Populate world with tiles
        self.tiles = []
        for i in range(self.length):
            space = Tile()
            space.x = i // self.X
            space.y = i % self.X
            self.tiles.append(space)
        self.tiles = np.asarray(self.tiles)
        self.tiles = self.tiles.reshape(self.dim)
        ### Generate Food & Cluster
        self.clusters = clusters
        self.cdev = cdev
        self.newcenterstep = self.timesteps/newcenterstep
        self.centers = np.random.choice(self.X, size=(1,self.clusters))
        self.center_box = (0, self.X) # defines the box that cluster centres are allowed to be in
        self.center_dev = self.X/self.cdev # defines the standard deviation of clusters
        ## Add food and CAs
        self.enz = enz
        for cell in self.seedCAs:
            newcell = CA(self)
            newcell.x = cell.x - self.xmin -1
            newcell.y = cell.y - self.ymin -1
            newcell.fill = cell.fill
            self.CAs.append(newcell) 
        seedx = 0
        for i in self.seed:
            seedy = 0
            for j in i:
                if j == (self.G0 + 1):
                    til = self.tiles[seedx,seedy]
                    til.CAfood = True
                    til.foodtough = self.foodtough
                    til.ftcap = til.foodtough
                seedy = seedy + 1
            seedx = seedx + 1
        ## World Output Data
        self.arrays = list()
        self.data = list()
        self.run = list()
        self.wants = list()
        self.display = display
    
    def __str__(self):
        return 'CA List: %s' % (self.CAs)
    
    def foodwrapper(self, value, dim):
        if dim == 0:
            xlim = self.X - 1
            if value > xlim:
                value = value - self.X
            elif value < 0:
                value = value + self.X
            else:
                value = value
        else:
            ylim = self.Y - 1
            if value > ylim:
                value = value - self.Y
            elif value < 0:
                value = value + self.Y
            else:
                value = value
        return(value)
    
    def wrapper(self, coordlist):
        xlim = self.X - 1
        ylim = self.Y - 1
        finallist = []
        for i in range(len(coordlist)):
            x = coordlist[i][0]
            y = coordlist[i][1]
            if x > xlim:
                x = x - self.X
            else:
                x = x
            if x < 0:
                x = x + self.X
            else:
                x = x
            if y > ylim:
                y = y - self.Y
            else:
                y = y
            if y < 0:
                y = y + self.Y
            else:
                y = y
            finallist.append((x,y))
        return finallist
    
    def showZoo(self):
        array = np.zeros((self.X, self.Y))
        tilearray = self.tiles.flatten()
        for til in tilearray:
            locx = til.x
            locy = til.y
            if til.CA == True:
                ## WHITE
                array[locx,locy] = self.G0 + 9
            elif til.CAfood == True:
                ## TEAL + BLUES
                array[locx,locy] = self.G0 + 1
            else:
                array[locx,locy] = til.CAgrad
        ## Save array frame
        self.arrays.append(array)
        ## Display the array
        if self.display == True:
            figure(figsize=(8, 8), dpi=70)
            pyplot.imshow(array, interpolation='nearest', cmap='gist_earth',
                          vmin = 0, vmax = self.G0 + 5)
            pyplot.show()
    
    ####################################
    ##  CHEMO GRADIENT FUNCTIONS
    ####################################
    
    def firstDegree(self, x, y):
        coordlist = [(x+1,y), (x,y+1), (x-1,y), (x,y-1)]
        coordlist = self.wrapper(coordlist)
        fList = []
        for i in coordlist:
            fList.append(self.tiles[i[0]][i[1]])
        return fList
    
    def secondDegree(self, x, y):
        coordlist = [(x+1,y+1), (x-1,y+1), (x-1,y-1), (x+1,y-1),
                     (x+2,y), (x,y+2), (x-2,y), (x,y-2)]
        coordlist = self.wrapper(coordlist)
        sList = []
        for i in coordlist:
            sList.append(self.tiles[i[0]][i[1]])
        return sList
    
    def thirdDegree(self, x, y):
        coordlist = [(x+2,y+1), (x-2,y+1), (x-2,y-1), (x+2,y-1),
                     (x+1,y+2), (x-1,y+2), (x-1,y-2), (x+1,y-2),
                     (x+3,y), (x,y+3), (x-3,y), (x,y-3)]
        coordlist = self.wrapper(coordlist)
        tList = []
        for i in coordlist:
            tList.append(self.tiles[i[0]][i[1]])
        return tList
    
    def fourthDegree(self, x, y):
        coordlist = [(x+3,y+1), (x-3,y+1), (x-3,y-1), (x+3,y-1),
                     (x+1,y+3), (x-1,y+3), (x-1,y-3), (x+1,y-3),
                     (x+2,y+2), (x-2,y+2), (x-2,y-2), (x+2,y-2),
                     (x+4,y), (x,y+4), (x-4,y), (x,y-4)]
        coordlist = self.wrapper(coordlist)
        fList = []
        for i in coordlist:
            fList.append(self.tiles[i[0]][i[1]])
        return fList
    
    def fifthDegree(self, x, y):
        coordlist = [(x+3,y+2), (x-3,y+2), (x-3,y-2), (x+3,y-2),
                     (x+2,y+3), (x-2,y+3), (x-2,y-3), (x+2,y-3),
                     (x+4,y+1), (x-4,y+1), (x-1,y-4), (x+1,y-4),
                     (x+1,y+4), (x-1,y+4), (x-4,y-1), (x+4,y-1),
                     (x+5,y), (x,y+5), (x-5,y), (x,y-5)]
        coordlist = self.wrapper(coordlist)
        fiList = []
        for i in coordlist:
            fiList.append(self.tiles[i[0]][i[1]])
        return fiList
    
    def newCenters(self):
        if self.timesteps / self.newcenterstep == self.timesteps // self.newcenterstep:
            self.centers = np.random.choice(self.X, size=(1,self.clusters))
            print("NEW CENTERS")
        
    def addFood(self):
        count = int(self.newfood)
        CAFindsX, y = make_blobs(n_samples=count, n_features=2, centers=self.centers, center_box=self.center_box, cluster_std=self.center_dev)
        CAFindsX = np.ndarray.tolist(CAFindsX)
        flatX = []
        for i in CAFindsX:
            for j in i:
                flatX.append(j)
        flatX = [round(elem) for elem in flatX]
        flatX = [self.foodwrapper(elem,0) for elem in flatX]    
        CAFindsY, y = make_blobs(n_samples=count, n_features=2, centers=self.centers, center_box=self.center_box, cluster_std=self.center_dev)
        CAFindsY = np.ndarray.tolist(CAFindsY)
        flatY = []
        for i in CAFindsY:
            for j in i:
                flatY.append(j)
        flatY = [round(elem) for elem in flatY]
        flatY = [self.foodwrapper(elem,0) for elem in flatY]
        for i in range(len(flatX)):
            til = self.tiles[flatX[i],flatY[i]]
            if til.CA == False:
                if til.CAfood == False:
                    chance = random.uniform(0,1)
                    if chance <= self.foodchance:
                        til.CAfood = True
                        til.foodtough = self.foodtough
                        til.foodvalue = til.foodtough * self.nutrition
                        til.ftcap = til.foodtough
        self.updateCAFGrad()
        self.updateCAFTough()
        
    ####################################
    ##  UPDATE TILE FUNCTIONS
    ####################################
    
    def updateCAGrad(self):
        for row in self.tiles:
            for tile in row:
                tile.CAgrad = 0
        CAx = []
        CAy = []
        for cell in self.CAs:
            CAx.append(cell.x)
            CAy.append(cell.y)
        for i in range(len(CAx)):
            posx = CAx[i]
            posy = CAy[i]
            ## Identity
            self.tiles[posx][posy].CAgrad = self.G0
            ## First Degree
            fList = self.firstDegree(posx,posy)
            for tile in fList:
                tile.CAgrad = tile.CAgrad + self.G1
            ## Second Degree
            sList = self.secondDegree(posx,posy)
            for tile in sList:
                tile.CAgrad = tile.CAgrad + self.G2
            ## Third Degree
            tList = self.thirdDegree(posx,posy)
            for tile in tList:
                tile.CAgrad = tile.CAgrad + self.G3
            fList = self.fourthDegree(posx,posy)
            for tile in fList:
                tile.CAgrad = tile.CAgrad + self.G4
            fiList = self.fifthDegree(posx,posy)
            for tile in fiList:
                tile.CAgrad = tile.CAgrad + self.G5
                
    def updateCAFGrad(self):
        for row in self.tiles:
            for tile in row:
                tile.CAFgrad = 0
                tile.desire = 4
        for i in range(self.X):
            for j in range(self.Y):
                tile = self.tiles[i][j]
                posx = tile.x
                posy = tile.y
                if tile.CAfood == True:
                    ## Identity
                    self.tiles[posx][posy].CAFgrad = self.G0
                    ## First Degree
                    fList = self.firstDegree(posx,posy)
                    for tile in fList:
                        tile.CAFgrad = tile.CAFgrad + self.G1
                    ## Second Degree
                    sList = self.secondDegree(posx,posy)
                    for tile in sList:
                        tile.CAFgrad = tile.CAFgrad + self.G2
                    ## Third Degree
                    tList = self.thirdDegree(posx,posy)
                    for tile in tList:
                        tile.CAFgrad = tile.CAFgrad + self.G3
                    ## Fourth Degree
                    fList = self.fourthDegree(posx,posy)
                    for tile in fList:
                        tile.CAFgrad = tile.CAFgrad + self.G4
                    ## Fifth Degree
                    fiList = self.fifthDegree(posx,posy)
                    for tile in fiList:
                        tile.CAFgrad = tile.CAFgrad + self.G5
                else:
                    tile.CAfood = False
                
    def updateCA(self):
        for row in self.tiles:
            for tile in row:
                tile.CA = False
        for cell in self.CAs:
            posx = cell.x
            posy = cell.y
            self.tiles[posx][posy].CA = True
            
    def updateCAFTough(self):
        foodcount = 0
        for row in self.tiles:
            for tile in row:
                if tile.CAfood == True:
                    foodcount = foodcount + 1
                    tile.foodtough = tile.foodtough + self.foodregenz
                    if tile.foodtough > tile.ftcap:
                        tile.foodtough = tile.ftcap
                    else: 
                        tile.foodtough = tile.foodtough
                else:
                    tile.CAfood = False
        self.CAfood = foodcount
                    
    ####################################
    ##  CHECK CONDITION FUNCTIONS
    ####################################

    def Reproduce(self, cell):
        neighbors = cell.neighbors
        check = 0
        while check == 0:
            if len(neighbors) == 0:
                check = 1
            else:
                loc = random.choice(neighbors)
                if loc.CA == True or loc.CAfood == True:
                    neighbors.remove(loc)
                else:
                    cell.children = cell.children + 1
                    cell.fill = cell.fill/2
                    self.tiles[loc.x][loc.y].CA = True
                    newcell = CA(self)
                    newcell.x = loc.x
                    newcell.y = loc.y
                    newcell.getNeighbors(self)
                    newcell.fill = cell.fill
                    self.CAs.append(newcell)
                    check = 1
                
    def checkEnd(self):
        if len(self.CAs) == 0:
            print("All synthetics have died.")
            return True
        elif self.steps == (self.timesteps - 1):
            self.step()
            print("End Zoo")
            return True
        else:
            return False
        
    ####################################
    ##  STEP FUNCTION
    ####################################
        
    def step(self):
        self.showZoo()
        self.addFood()
        
        ### EAT & ADD FOOD
        for cell in self.CAs:
            ## Each cell eats and update chemical gradients (in function)
            cell.eatZoo(self)
        
        ### MOVE
        self.CAs.sort()
        for cell in self.CAs:
            cell.move_desires(self)
            cell.move_adhesion(self)
            ## Move each cell and update chemical gradiants (in function)
            ## Note that for CAS there are two move functions to choose from,
            ## comment out one.
            ## cell.movebrain(self)
            cell.movesimple(self)
        
        ### STEP
        nUni = len(self.CAs)
        self.run.append([self.steps,nUni,self.CAfood])
        self.steps = self.steps + 1
    

########################################
########################################
##  UNICELLULARS
########################################                        
########################################
        
class CA:
    def __init__(self,env):
         ## BASICS
         self.ID = len(env.CAs) + 1
         self.start = env.steps
         self.lifespan = env.steps
         self.end = 0
         self.x = random.randrange(0,(env.X-1),1)
         self.y = random.randrange(0,(env.Y-1),1)
         ## EATING & MOVING
         self.enz = env.enz
         self.fill = 0.8
         self.neighbors = []
         self.children = 0
         self.motors = []
         self.spaces = [False,False,False,False,False]
         self.goodmoves = []
         self.movedesire = 4
         self.food_wants = list()
         self.friend_wants = list()
         self.desire_wants = list()
         self.choices = list()
         
    def __str__(self):
        return 'ID: %s. Lifespan: %s. Fill: %s.' % (self.ID, self.lifespan, self.fill)
    
    def __lt__(self, other):
         return self.fill < other.fill
    
    def getNeighbors(self, env):
        x = self.x
        y = self.y
        spaces = [False,False,False,False,False]
        ## ncl = neighbor coord list
        ncl = [(x+1,y), (x,y+1),
               (x-1,y), (x,y-1)]
        ncl = env.wrapper(ncl)
        neighborlist = []
        for coord in ncl:
            neighborlist.append(env.tiles[coord[0]][coord[1]])
        for neighbor in neighborlist:
            if neighbor.CA == True or neighbor.CAfood == True:
                spaces[neighborlist.index(neighbor)] = True
        self.neighbors = neighborlist
        self.spaces = spaces
            
    ####################################
    ##  MOVEMENT FUNCTIONS
    ####################################
        
    def moveup(self):
        self.y = self.y+1
    def movedown(self):
        self.y = self.y-1
    def moveleft(self):
        self.x = self.x-1
    def moveright(self):
        self.x = self.x+1
    def stay(self):
        self.x = self.x
        
    def breaker(self, env):
        self.getNeighbors(env)
        count = 0
        result = False
        for n in self.neighbors:
            if n.CAfood == True:
                result = True
                break
            elif n.CA == True:
                count = count + 1
        if count == 4:
            result = True
        return result
        
    def getGoodMoves(self, env):
        env.updateCA()
        self.goodmoves = [4]
        maxmot = max(self.motors)
        bestvalue = -1000
        while maxmot >= bestvalue:
            result = self.breaker(env)
            if result == True:
                break
            else:
                move = self.motors.index(maxmot)
                if self.spaces[move] == True:
                    self.motors[move] = -10
                    maxmot = max(self.motors)
                else:
                    self.goodmoves.append(move)
                    bestvalue = maxmot
                    self.motors[move] = -10
                    maxmot = max(self.motors)
        if len(self.goodmoves) > 1:
            self.goodmoves.remove(4)
        return(self.goodmoves)
    
    def move_desires(self, env):
        env.updateCA()
        self.getNeighbors(env)
        self.motors = []
        ## FOOD RULE
        for tile in self.neighbors:
            food_want = tile.CAFgrad*env.caf_scal
            self.food_wants.append(food_want)
            self.motors.append(food_want)
        
        ### NEIGHBOR RULE
        env.updateCAGrad()
        grads = []
        for tile in self.neighbors:
            ## We subtract 0.5 to scale negative and positive, so full agents
            ## are less inclined to approach other agents like them. Multiply 
            ## by 2 to scale it back to the fill's original value (for 
            ## proportioning the constants)
            #grads.append(sigmoid(tile.CAgrad*zoo.agent_chemo_scal*(self.fill - 0.5)))
            grads.append(tile.CAgrad*env.agent_chemo_scal*-2*(self.fill - 0.5))
        friend_want = grads
        self.friend_wants.append(friend_want)
        self.motors = np.add(self.motors,grads)
        self.motors = self.motors.tolist()
        
        self.movedesire = self.motors.index(max(self.motors))
        env.tiles[self.x][self.y].desire = self.movedesire
        
    def move_adhesion(self, env):
        self.getNeighbors(env)
        desires1 = []
        for n in self.neighbors:
            if n.CA == True:
                desires1.append(n.desire)
            else:
                desires1.append(10)
        for i in range(3):
            if desires1[i] <= 3:
                d = desires1[i]
                ### ADHESION RULE
                #self.motors[d] = self.motors[d] + sigmoid(zoo.adhesion_scal * (1/self.fill))
                desire_want = [d,(env.adhesion_scal * (0.5/self.fill))]
                self.desire_wants.append(desire_want)
                self.motors[d] = self.motors[d] + (env.adhesion_scal * (0.5/self.fill))
            
    def movesimple(self,env):
        env.tiles[self.x][self.y].fill = 0
        #self.move_agent_desires(zoo)
        self.goodmoves = self.getGoodMoves(env)
        move = random.choice(self.goodmoves)
        self.choices.append(move)
        if move == 0:
            self.moveright()
        elif move == 1:
            self.moveup()
        elif move == 2:
            self.moveleft()
        elif move == 3:
            self.movedown()
        elif move == 4:
            self.stay()
        coords = [(self.x, self.y)]
        coords = env.wrapper(coords)
        self.x = coords[0][0]
        self.y = coords[0][1]
        env.tiles[self.x][self.y].fill = self.fill
        self.motors = []
        self.fill = self.fill - foodloss
        env.updateCA()
        env.updateCAGrad()
        self.getNeighbors(env)
    
    ####################################
    ##  EATING FUNCTIONS
    ####################################
    
    ## There has to be a cleaner way of doing this but I just need the zoo to 
    ## not change their fill values
     
    def eatZoo(self, env):
        self.getNeighbors(env)
        for neighbor in self.neighbors:
            if neighbor.CAfood == True:
                posx = neighbor.x
                posy = neighbor.y
                env.tiles[posx][posy].foodtough = env.tiles[posx][posy].foodtough - self.enz
                if env.tiles[posx][posy].foodtough <= 0:
                    env.tiles[posx][posy].CAfood = False
                    env.tiles[posx][posy].foodvalue = 0.0
                    env.tiles[posx][posy].foodtough = 0.0
                    env.tiles[posx][posy].ftcap = 0.0
                
    def eatWorld(self, world):
        self.getNeighbors(world)
        for neighbor in self.neighbors:
            if neighbor.CAfood == True:
                posx = neighbor.x
                posy = neighbor.y
                world.tiles[posx][posy].foodtough = world.tiles[posx][posy].foodtough - self.enz
                if world.tiles[posx][posy].foodtough <= 0:
                    self.fill = self.fill + world.tiles[posx][posy].foodvalue
                    world.tiles[posx][posy].CAfood = False
                    world.tiles[posx][posy].foodvalue = 0.0
                    world.tiles[posx][posy].foodtough = 0.0
                    world.tiles[posx][posy].ftcap = 0.0
                else:
                    self.fill = self.fill
            else:
                self.fill = self.fill
        
        
        
        
        
        
        
    