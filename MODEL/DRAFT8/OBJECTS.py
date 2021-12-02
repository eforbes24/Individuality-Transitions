#### MULTICELLULAR AUTOMATA MODEL ####
## Eden Forbes
## Object Class Definitions

#### LIBRARIES
import numpy as np
import random
import statistics as st
from sklearn.datasets.samples_generator import make_blobs
from matplotlib import pyplot
from matplotlib.pyplot import figure

#### OBJECTS

########################################
########################################
##  TILES
########################################                        
########################################

class Tile:
    CA = False
    IND = False
    CAfood = False
    CAgrad = 0.0
    CAFgrad = 0.0
    foodtough = 0.0
    ftcap = 0.0
    foodvalue = 0.0
    x = 0
    y = 0

########################################
########################################
##  WORLD
########################################                        
########################################
    
class World:
    def __init__(self, timesteps, X, Y, start_CA, CAfood, clusters, cdev, 
                 foodregenz, newcenterstep, g0, g1, g2, g3, g4, g5, 
                 foodtoughlow, foodtoughhigh, nutrition,
                 lowfillstart, highfillstart, foodloss, newfood, 
                 foodchance, paniclevel, dissolve_thresh, diffusion,
                 enz, diffenz, germenz, display):
        ## Give world attributes
        self.timesteps = timesteps
        self.X = X
        self.Y = Y
        self.foodloss = foodloss
        self.newfood = newfood
        self.foodchance = foodchance
        self.steps = 0
        self.CAs = list()
        self.Multis = list()
        self.paniclevel = paniclevel
        self.dissolve_thresh = dissolve_thresh
        self.diffusion = diffusion
        ## Gradient & Enzyme attributes
        self.G0 = g0
        self.G1 = g1
        self.G2 = g2
        self.G3 = g3
        self.G4 = g4
        self.G5 = g5
        self.foodregenz = foodregenz
        self.ftl = foodtoughlow
        self.fth = foodtoughhigh
        self.nutrition = nutrition
        self.lowfillstart = lowfillstart
        self.highfillstart = highfillstart
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
        self.clusters = clusters
        self.cdev = cdev
        self.newcenterstep = self.timesteps/newcenterstep
        self.centers = np.random.choice(self.X, size=(1,self.clusters))
        self.center_box = (0, self.X) # defines the box that cluster centres are allowed to be in
        self.center_dev = self.X/self.cdev # defines the standard deviation of clusters
        CAFinds, y = make_blobs(n_samples=CAfood, n_features=2, centers=self.centers, center_box=self.center_box, cluster_std=self.center_dev)
        CAFinds = np.ndarray.tolist(CAFinds)
        for i in CAFinds:
            i[0] = round(i[0])
            i[1] = round(i[1])
        CAFinds = self.wrapper(CAFinds)
        for i in CAFinds:
            til = self.tiles[i[0],i[1]]
            til.CAfood = True
            til.foodtough = random.uniform(self.ftl, self.fth)
            til.ftcap = til.foodtough
        ## Generate CAs
        self.enz = enz
        self.diffenz = diffenz
        self.germenz = germenz
        for i in range(start_CA):
            cell = CA(self)
            self.CAs.append(cell)
        ## World Output Data
        self.arrays = list()
        self.data = list()
        self.display = display
    
    def __str__(self):
        return 'CA List: %s' % (self.CAs)
    
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
        for multi in self.Multis:
            for cell in multi.CAs:
                locx = cell.x
                locy = cell.y
                if cell.diff == True:
                    ## OLIVE GREEN
                    array[locx][locy] = self.G0 + 5
                elif cell.germ == True:
                    ## ORANGE
                    array[locx][locy] = self.G0 + 7
        ## Save array frame
        self.arrays.append(array)
        ## Display the array
        if self.display == True:
            figure(figsize=(8, 8), dpi=70)
            pyplot.imshow(array, interpolation='nearest', cmap='gist_earth')
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
        count = self.newfood
        while count > 0:
            CAFinds, y = make_blobs(n_samples=count, n_features=2, centers=self.centers, 
                                    center_box=self.center_box, cluster_std=self.center_dev)
            CAFinds = np.ndarray.tolist(CAFinds)
            for i in CAFinds:
                i[0] = round(i[0])
                i[1] = round(i[1])
            CAFinds = self.wrapper(CAFinds)
            for CAF in CAFinds:
                til = self.tiles[CAF[0],CAF[1]]
                if til.CA == False:
                    if til.CAfood == False:
                        if til.IND == False:
                            chance = random.uniform(0,1)
                            if chance < self.foodchance:
                                til.CAfood = True
                                til.foodtough = random.uniform(self.ftl, self.fth)
                                til.foodvalue = til.foodtough * self.nutrition
                                til.ftcap = til.foodtough
                                count = 0
                            else:
                                count = 0
                    if count == 0:
                        break
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
        for multi in self.Multis:
            for cell in multi.CAs:
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
            
    def updateIND(self):
        for row in self.tiles:
            for tile in row:
                tile.IND = False
        for multi in self.Multis:
            for cell in multi.CAs:
                posx = cell.x
                posy = cell.y
                self.tiles[posx][posy].IND = True
            
    def updateCAFTough(self):
        for row in self.tiles:
            for tile in row:
                if tile.CAfood == True:
                    tile.foodtough = tile.foodtough + self.foodregenz
                    if tile.foodtough > tile.ftcap:
                        tile.foodtough = tile.ftcap
                    else: 
                        tile.foodtough = tile.foodtough
                else:
                    tile.CAfood = False
                    
    def updateGenomes(self):
        for cell in self.CAs:
            cell.genome[1] = cell.fill
            addFood = 0
            for neighbor in cell.neighbors:
                addFood = addFood + neighbor.CAFgrad
            cell.genome[2] = addFood
            addCell = 0
            for neighbor in cell.neighbors:
                addCell = addCell + neighbor.CAgrad
            cell.genome[3] = addCell
            cell.genome[4] = random.uniform(0,1)
                    
    ####################################
    ##  CHECK CONDITION FUNCTIONS
    ####################################
    
    ### Genome is made of:
    ### [0] = State
    ### [1] = Fill
    ### [2] = FoodGrad
    ### [3] = CellGrad
    ### [4] = RNG
    
    def checkDifferentiate(self, cell_A, cell_B):
        gen1 = cell_A.genome
        gen2 = cell_B.genome
        if gen1[0] == gen2[0]:
            if gen1[1] == gen2[1]:
                if gen1[2] == gen2[2]:
                    if gen1[3] == gen2[3]:
                        if gen1[4] < gen2[4]:
                            return cell_A
                        else:
                            return cell_B
                    elif gen1[3] < gen2[3]:
                        return cell_A
                    else: 
                        return cell_B
                elif gen1[2] < gen2[2]:
                    return cell_A
                else:
                    return cell_B
            elif gen1[1] < gen2[1]:
                return cell_A
            else:
                return cell_B
        elif (gen1[0] == 0 and gen2[0] == 1) or (gen1[0] == 0 and gen2[0] == 2):
            return cell_A
            
        elif (gen1[0] == 1 and gen2[0] == 0) or (gen1[0] == 2 and gen2[0] == 0):
            return cell_B
            
    
    def Reproduce(self, cell, multi, multcon):
        neighbors = cell.neighbors
        check = 0
        if multcon == 0:
            while check == 0:
                if len(neighbors) == 0:
                    check = 1
                else:
                    loc = random.choice(neighbors)
                    if loc.CA == True or loc.IND == True or loc.CAfood == True:
                        neighbors.remove(loc)
                    else:
                        cell.children = cell.children + 1
                        cell.fill = cell.fill/2
                        self.tiles[loc.x][loc.y].CA = True
                        newcell = CA(self)
                        newcell.x = loc.x
                        newcell.y = loc.y
                        newcell.getNeighbors(self)
                        mutation = np.random.normal(0,0.1,size=(4,4))
                        newcell.brain = cell.brain + mutation
                        newcell.fill = cell.fill
                        self.CAs.append(newcell)
                        check = 1
        elif multcon == 1:
            while check == 0:
                if len(neighbors) == 0:
                    check = 1
                else:
                    loc = random.choice(neighbors)
                    if loc.CA == True or loc.IND == True or loc.CAfood == True:
                        neighbors.remove(loc)
                    else:
                        cell.children = cell.children + 1
                        cell.fill = cell.fill/2
                        self.tiles[loc.x][loc.y].IND = True
                        newcell = CA(self)
                        newcell.x = loc.x
                        newcell.y = loc.y
                        newcell.getNeighbors(self)
                        mutation = np.random.normal(0,0.1,size=(4,4))
                        newcell.brain = cell.brain + mutation
                        newcell.fill = cell.fill
                        newcell.differentiate(2,self)
                        multi.CAs.append(newcell)
                        check = 1
                    
    def checkRepro(self):
        for cell in self.CAs:
            cell.getNeighbors(self)
            if cell.fill >= 1:
                self.Reproduce(cell, [], 0)
                #print("A cell reproduced!")
            else:
                self.CAs = self.CAs
        for multi in self.Multis:
            for cell in multi.CAs:
                cell.getNeighbors(self)
                if cell.genome[0] == 2:
                    if cell.fill >= 1:
                        neighbors = cell.neighbors
                        check = 0
                        while check == 0:
                            if len(neighbors) == 0:
                                check = 1
                            else:
                                loc = random.choice(neighbors)
                                if loc.IND == True or loc.CA == True:
                                    neighbors.remove(loc)
                                else:
                                    self.Reproduce(cell, multi, 1)
                                    multi.getNeighborsIND(self)
                                    check = 1
        self.updateCA()
        self.updateIND()
    
    def AggregateCA(self, cellA, cellB):
        newind = Multi(self)
        diffcell = self.checkDifferentiate(cellA, cellB)
        if diffcell == cellB:
            cellA.differentiate(2,self)
            cellB.differentiate(1,self)
        else:
            cellA.differentiate(1,self)
            cellB.differentiate(2,self)
        newind.CAs.append(cellA)
        newind.CAs.append(cellB)
        self.Multis.append(newind)
        self.tiles[cellA.x][cellA.y].IND = True
        self.tiles[cellB.x][cellB.y].IND = True
        self.CAs.remove(cellA)
        self.CAs.remove(cellB)
        self.tiles[cellA.x][cellA.y].CA = False
        self.tiles[cellB.x][cellB.y].CA = False
    
    def checkCAAggregate(self):
        for cell in self.CAs:
            if cell.fill < self.paniclevel:
                cell.getNeighbors(self)
                for neighbor in cell.neighbors:
                    cell2 = []
                    multicheck = False
                    if neighbor.CA == True:
                        coords = self.wrapper([(neighbor.x,neighbor.y)])
                        posx = coords[0][0]
                        posy = coords[0][1]
                        rng = len(self.CAs)
                        for i in (range(rng)):
                            if self.CAs[i].x == posx and self.CAs[i].y == posy:
                                cell2 = self.CAs[i]
                                break
                        if cell2.fill < self.paniclevel:
                            self.AggregateCA(cell,cell2)
                    if cell2 != []:
                        break
                    elif neighbor.IND == True:
                        coords = self.wrapper([(neighbor.x,neighbor.y)])
                        posx = coords[0][0]
                        posy = coords[0][1]
                        for multi in self.Multis:
                            for CA in multi.CAs:
                                CAcheck = False
                                if CA.x == posx and CA.y == posy:
                                    if CA.fill < self.paniclevel:
                                        cell.differentiate(1,self)
                                        multi.CAs.append(cell)
                                        self.tiles[cell.x][cell.y].CA = False
                                        self.tiles[cell.x][cell.y].IND = True
                                        multicheck = True
                                        CAcheck = True
                                        self.CAs.remove(cell)
                                        self.updateCA()
                                if CAcheck == True:
                                    break
                            if multicheck == True:
                                break
                    if multicheck == True:
                        break
        self.updateCA()
        self.updateIND()
    
    def checkINDAggregate(self):
        for multi in self.Multis:
            multi.getNeighborsIND(self)
            for n in multi.tnl:
                if n.IND == True:
                    dex = multi.tnl.index(n)
                    cell = multi.whose[dex]
                    if cell.fill < self.paniclevel:
                        for m in self.Multis:
                            for c in m.CAs:
                                if n.x == c.x and n.y == c.y:
                                    dex1 = m.CAs.index(c)
                                    dex2 = self.Multis.index(m)
                                    cell2 = self.Multis[dex2].CAs[dex1]
                                    break
                        if cell2.fill < self.paniclevel:
                            for i in range(len(self.Multis[dex2].CAs)):
                                multi.CAs.append(self.Multis[dex2].CAs[i])
                        self.Multis.remove(self.Multis[dex2])
                        multi.getNeighborsIND(self)
                        self.updateIND()
                        break
    
    def checkDeath(self):
        for cell in self.CAs:
            if cell.fill <= 0:
                cell.end = self.steps
                self.data.append(["cell", cell.ID, cell.start, cell.end, 
                                  cell.lifespan, cell.children, cell.brain])
                posx = cell.x
                posy = cell.y
                self.tiles[posx][posy].CA = False
                self.CAs.remove(cell)
              #  print("A cell has died :(")
            else:
                cell.lifespan = cell.lifespan + 1
        for multi in self.Multis:
            check = 0
            for cell in multi.CAs:
                if cell.fill <= 0:
                    check = 1
                    multi.end = self.steps
                    for cell in multi.CAs:
                        posx = cell.x
                        posy = cell.y
                        self.tiles[posx][posy].IND = False
                    self.data.append(["multi", "died", multi.ID, multi.start, 
                                      multi.end, multi.lifespan, multi.size, 
                                      list(multi.CAs)])
                    self.Multis.remove(multi)
                    break
                    #  print("A multi has died :(")
                else:
                    cell.fill = cell.fill
            if check == 0:
                multi.lifespan = multi.lifespan + 1
        self.updateCA()
        self.updateIND()
                
    def checkEnd(self):
        if len(self.CAs) == 0 and len(self.Multis) == 0:
            print("All beings have died.")
            return True
        else:
            return False
        
    ####################################
    ##  STEP FUNCTION
    ####################################
        
    def step(self):
        self.showWorld()
        
        ### EAT & ADD FOOD
        for multi in self.Multis:
            ## Each multi eats and update chemical gradients (in function)
            multi.multieat(self)
        for cell in self.CAs:
            ## Each cell eats and update chemical gradients (in function)
            cell.eat(self)
        self.addFood()
        
        ### MOVE
        for multi in self.Multis:
            ## Move each multi and update chemical gradiants (in function)
            multi.movemulti(self)
        for cell in self.CAs:
            ## Move each cell and update chemical gradiants (in function)
            ## Note that for CAS there are two move functions to choose from,
            ## comment out one.
            ## cell.movebrain(self)
            cell.movesimple(self)
        
        ### CHECKS
        ## Check for cell death, aggregation, and reproduction
        self.checkDeath()
        self.checkCAAggregate()
        self.checkINDAggregate()
        self.checkRepro()
        
        ### STEP
        self.steps = self.steps + 1

########################################
########################################
##  UNICELLULARS
########################################                        
########################################
        
class CA:
    def __init__(self,world):
         ## BASICS
         self.ID = len(world.CAs) + 1
         self.start = world.steps
         self.lifespan = world.steps
         self.end = 0
         self.x = random.randrange(0,(world.X-1),1)
         self.y = random.randrange(0,(world.Y-1),1)
         ## DIFFERENTIATON
         self.genome = np.zeros(5)
         self.diff = False
         self.germ = False
         ## EATING & MOVING
         self.enz = world.enz
         self.fill = random.uniform(world.lowfillstart, world.highfillstart)
         self.neighbors = []
         self.children = 0
         self.spaces = [False,False,False,False,False]
         ## BRAIN STUFF
         self.brain = np.random.uniform(-1,1,size=(4,4))
         
         
    def __str__(self):
        return 'ID: %s. Lifespan: %s. Fill: %s.' % (self.ID, self.lifespan, self.fill)
    
    def getNeighbors(self, world):
        x = self.x
        y = self.y
        spaces = [False,False,False,False,False]
        ## ncl = neighbor coord list
        ncl = [(x+1,y), (x,y+1),
               (x-1,y), (x,y-1)]
        ncl = world.wrapper(ncl)
        neighborlist = []
        for coord in ncl:
            neighborlist.append(world.tiles[coord[0]][coord[1]])
        for neighbor in neighborlist:
            if neighbor.CA == True or neighbor.CAfood == True or neighbor.IND == True:
                spaces[neighborlist.index(neighbor)] = True
        self.neighbors = neighborlist
        self.spaces = spaces
        
    ### STATES
    ### 0 = Undifferentiated
    ### 1 = Differentiated Cell
    ### 2 = Germ Cell
    
    def differentiate(self, state, world):
        if state == 0:
            self.diff = False
            self.germ = False
            self.enz = world.enz
            self.genome[0] = 0
        elif state == 1:
            self.diff = True
            self.germ = False
            self.enz = world.diffenz
            self.genome[0] = 1
        elif state == 2:
            self.diff = False
            self.germ = True
            self.enz = world.germenz
            self.genome[0] = 2
            
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
        
#    def movebrain(self, world):
#        self.getNeighbors(world)
#        sensors = []
#        for tile in self.neighbors:
#            sensors.append(tile.CAFgrad)
#        spaces = self.checkFill(world)
#        motors = np.dot(sensors, self.brain)
#        motors = motors.tolist()
#        checker = 0
#        goodmoves = []
#        while checker == 0:
#            maxmot = max(motors)
#            if maxmot > 0 or maxmot < 0:
#                move = motors.index(maxmot)
#                if spaces[move] == True:
#                    motors[move] = -10
#                    checker = 0
#                else:
#                    goodmoves.append(move)
#                    checker = 1
#            else:
#                while 0 in motors:
#                    zeroplace = motors.index(0)
#                    goodmoves.append(zeroplace)
#                    motors[zeroplace] = -10
#                checker = 1
#        move = random.choice(goodmoves)
#        if move == 0:
#            self.moveright()
#        elif move == 1:
#            self.moveup()
#        elif move == 2:
#            self.moveleft()
#        elif move == 3:
#            self.movedown()
#        elif move == 4:
#            self.stay()
#        coords = [(self.x, self.y)]
#        coords = world.wrapper(coords)
#        self.x = coords[0][0]
#        self.y = coords[0][1]
        
    def movesimple(self, world):
        world.updateCA()
        world.updateIND()
        self.getNeighbors(world)
        motors = []
        smeller = []
        spaces = self.spaces
        ## Two different sets of information used as input, depending on 
        ## if the cell is well-enough fed
        if self.fill > world.paniclevel:
            for tile in self.neighbors:
                motors.append(tile.CAFgrad)
            goodmoves = [4]
            maxmot = max(motors)
            bestvalue = 0
            while maxmot >= bestvalue:
                if maxmot >= world.G0:
                    break
                else:
                    move = motors.index(maxmot)
                    if spaces[move] == True:
                        motors[move] = -10
                        maxmot = max(motors)
                    else:
                        goodmoves.append(move)
                        bestvalue = maxmot
                        motors[move] = -10
                        maxmot = max(motors)
            if len(goodmoves) > 1:
                goodmoves.remove(4)
            move = random.choice(goodmoves)
        else:
            for tile in self.neighbors:
                motors.append(tile.CAgrad)
                smeller.append(tile.CAFgrad)
            goodmoves = [4]
            maxmot = max(motors)
            maxsmell = max(smeller)
            bestvalue = 0
            while maxmot >= bestvalue:
                if maxsmell >= world.G0 or maxmot >= world.G0:
                    break
                else:
                    move = motors.index(maxmot)
                    if spaces[move] == True:
                        motors[move] = -10
                        maxmot = max(motors)
                    else:
                        goodmoves.append(move)
                        bestvalue = maxmot
                        motors[move] = -10
                        maxmot = max(motors)
            if len(goodmoves) > 1:
                goodmoves.remove(4)
            move = random.choice(goodmoves)
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
        coords = world.wrapper(coords)
        self.x = coords[0][0]
        self.y = coords[0][1]
        world.updateCA()
        world.updateCAGrad()
        self.getNeighbors(world)
        if world.tiles[self.x][self.y].CA == True and world.tiles[self.x][self.y].IND == True:
            print("CHECK 1")
    
    ####################################
    ##  EATING FUNCTIONS
    ####################################
     
    def eat(self, world):
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
        self.fill = self.fill - world.foodloss
        
########################################
########################################
##  MULTICELLULARS
########################################                        
########################################
              
class Multi:
    def __init__(self,world):
        # BASICS
        self.ID = len(world.Multis) + 1
        self.start = world.steps
        self.lifespan = world.steps
        self.end = 0
        # INTERAL CELLS 
        self.CAs = list()
        self.size = len(self.CAs)
        # NEIGHBOR RELATIONS
        self.tnl = list()
        self.whose = list()
        self.tdl = list()
        self.tcg = [0,0,0,0]
        self.spaces = [False,False,False,False,False]
        
    def __str__(self):
        return 'ID: %s. Lifespan: %s. Size: %s' % (self.ID, self.lifespan, self.size)
    
    def checkFillIND(self,n,spaces,dex):
        if n.CA == True:
            #print("FULL: CA")
            spaces[dex] = True
        elif n.IND == True:
            #print("FULL: IND")
            spaces[dex] = True
        elif n.CAfood == True:
            #print("FULL: CA FOOD")
            spaces[dex] = True
        elif spaces[dex] == True:
            spaces[dex] = True
        else: 
            spaces[dex] == False       
        return spaces
        
    ## THIS FUNCTION IS STILL CAUSING PROBLEMS - ANY CUBE THINKS IT CAN ONLY GO RIGHT
    def getNeighborsIND(self, world):
        totalneighbors = []
        totaldirections = []
        totalCAFGrads = [0,0,0,0]
        whoseneighbor = []
        spaces = [False,False,False,False,False]
        CAlist = self.CAs
        for cell in self.CAs:
            neighborlist = []
            cell.getNeighbors(world)
            for n in cell.neighbors:
                neighborlist.append(n)
                for c in CAlist:
                    if n.x == c.x and n.y == c.y:
                        #print(n)
                        neighborlist.remove(n)
                        #print("REMOVED")
                        break
            for n in neighborlist:
                if n.x == cell.x+1 or (cell.x == world.X - 1 and n.x == 0):
                    #print("RIGHT")
                    totalneighbors.append(n)
                    whoseneighbor.append(cell)
                    totaldirections.append(0)
                    totalCAFGrads[0] = totalCAFGrads[0] + n.CAFgrad
                    spaces = self.checkFillIND(n,spaces,0)
                elif n.y == cell.y+1 or (cell.y == world.Y - 1 and n.y == 0):
                    #print("UP")
                    totalneighbors.append(n)
                    whoseneighbor.append(cell)
                    totaldirections.append(1)
                    totalCAFGrads[1] = totalCAFGrads[1] + n.CAFgrad
                    spaces = self.checkFillIND(n,spaces,1)
                elif n.x == cell.x-1 or (cell.x == 0 and n.x == world.X - 1):
                    #print("LEFT")
                    totalneighbors.append(n)
                    whoseneighbor.append(cell)
                    totaldirections.append(2)
                    totalCAFGrads[2] = totalCAFGrads[2] + n.CAFgrad
                    spaces = self.checkFillIND(n,spaces,2)
                elif n.y == cell.y-1 or (cell.y == 0 and n.y == world.Y - 1):
                    #print("DOWN")
                    totalneighbors.append(n)
                    whoseneighbor.append(cell)
                    totaldirections.append(3)
                    totalCAFGrads[3] = totalCAFGrads[3] + n.CAFgrad
                    spaces = self.checkFillIND(n,spaces,3)
                else:
                    print("ERROR")
#        for i in range(len(self.tnl)):
#            if self.tnl[i] in self.CAs:
#                self.tnl.remove(self.tnl[i])
#                self.whose.remove(self.whose[i])
#                self.tdl.remove(self.tdl[i])
        self.tnl = totalneighbors
        self.whose = whoseneighbor
        self.tdl = totaldirections
        self.tcg = totalCAFGrads
        self.spaces = spaces
            
  
    
    ####################################
    ##  MOVEMENT FUNCTIONS
    ####################################
    
    ### NEED TO ADD A MULTI BRAIN SYSTEM WHERE THE BRAIN GROWS PERCEPTUAL 
    ### INPUTS AS CELLS ARE ADDED
    
    ## OTHER MOVEMENT IDEAS (from Josh & Kory)
    ## Have cells emit signals to one another that play into their 
    ## chemotactic decisions
    
    ## Store the actions for each cell as decisions are made, then execute
    ## all at once (instead of moving one cell at a time: keep the current
    ## state of the cells and the state at t+1) --> should implement this
    ## up top too
    
    def direct(self, world):
        ## Here is where you do all the inbodied math for deciding where to go
        self.getNeighborsIND(world)
        motors = self.tcg
        spaces = self.spaces
        ## Then do the max math (best for now)
        goodmoves = [4]
        maxmot = max(motors)
        bestvalue = 0
        while maxmot >= bestvalue:
            if maxmot >= world.G0:
                break
            else:
                move = motors.index(maxmot)
                if spaces[move] == True:
                    motors[move] = - 10
                    maxmot = max(motors)
                else:
                    goodmoves.append(move)
                    bestvalue = maxmot
                    motors[move] = -10
                    maxmot = max(motors)
        if len(goodmoves) > 1:
            goodmoves.remove(4)
        move = random.choice(goodmoves)
        return(move)
        
    def movemulti(self, world):
        direct = self.direct(world)
        if direct == 0:
            for cell in self.CAs:
                #print("moved right")
                cell.moveright()
                coords = [(cell.x, cell.y)]
                coords = world.wrapper(coords)
                cell.x = coords[0][0]
                cell.y = coords[0][1]
        elif direct == 1:
            for cell in self.CAs:
                #print("moved up")
                cell.moveup()
                coords = [(cell.x, cell.y)]
                coords = world.wrapper(coords)
                cell.x = coords[0][0]
                cell.y = coords[0][1]
        elif direct == 2:
            for cell in self.CAs:
                #print("moved left")
                cell.moveleft()
                coords = [(cell.x, cell.y)]
                coords = world.wrapper(coords)
                cell.x = coords[0][0]
                cell.y = coords[0][1]
        elif direct == 3:
            for cell in self.CAs:
                #print("moved down")
                cell.movedown()
                coords = [(cell.x, cell.y)]
                coords = world.wrapper(coords)
                cell.x = coords[0][0]
                cell.y = coords[0][1]
        elif direct == 4:
            for cell in self.CAs:
                #print("stayed")
                cell.x = cell.x
                cell.y = cell.y
        world.updateIND()
        world.updateCAGrad()
        self.getNeighborsIND(world)
        for CA in self.CAs:
            if world.tiles[CA.x][CA.y].CA == True and world.tiles[CA.x][CA.y].IND == True:
                print("CHECK 2", CA.x, CA.y)
                
    ####################################
    ##  EATING FUNCTIONS
    ####################################
        
    def multieat(self, world):
        lent = len(self.tnl)
        for i in range(lent):
            neighbor = self.tnl[i]
            homeowner = self.whose[i]
            if neighbor.CAfood == True:
                posx = neighbor.x
                posy = neighbor.y
                world.tiles[posx][posy].foodtough = world.tiles[posx][posy].foodtough - homeowner.enz
                if world.tiles[posx][posy].foodtough <= 0:
                    homeowner.fill = homeowner.fill + world.tiles[posx][posy].foodvalue
                    ## USE BELOW FOR AUTOMATIC FOOD SHARING, NO DIFFUSION
#                    size = len(self.CAs)
#                    for cell in self.CAs:
#                        cell.fill = cell.fill + (world.tiles[posx][posy].foodvalue/size)
                    world.tiles[posx][posy].CAfood = False
                    world.tiles[posx][posy].foodvalue = 0.0
                    world.tiles[posx][posy].foodtough = 0.0
                    world.tiles[posx][posy].ftcap = 0.0
                    world.updateCAFGrad()
        for cell in self.CAs:
            cell.fill = cell.fill - world.foodloss
        self.multidissolve(world)
        self.multishare(world)
        
    def multishare(self, world):
        stomachs = []
        for cell in self.CAs:
            stomachs.append(cell.fill)
        avg_fill = st.mean(stomachs)
        for cell in self.CAs:
            if cell.fill > avg_fill:
                ## THIS EQUATION IS REALLY CRUDE BUT KEEPS FOOD LEVEL ==
                cell.fill = cell.fill - (avg_fill * world.diffusion)
            elif cell.fill == avg_fill:
                cell.fill = cell.fill
            else:
                cell.fill = cell.fill + (avg_fill * world.diffusion)
            
    def multidissolve(self, world):
        cellfills = []
        for CA in self.CAs:
            cellfills.append(CA.fill)
        if st.mean(cellfills) > world.dissolve_thresh:
            self.end = world.steps
            for cell in self.CAs:
                world.CAs.append(cell)
                world.tiles[cell.x][cell.y].IND = False
                world.tiles[cell.x][cell.y].CA = True
                cell.diff = False
                cell.germ = False
                cell.enz = world.enz
            world.data.append(["multi", "dissolved", self.ID, self.start, self.end, self.lifespan, self.size, list(self.CAs)])
            world.Multis.remove(self)
        else:
            cellfills = cellfills
         
        
