#### AUTOMATA MODEL ####
## Eden Forbes
## Object Class Definitions

#### LIBRARIES
import numpy as np
import random
from sklearn.datasets.samples_generator import make_blobs
from matplotlib import pyplot
from matplotlib.pyplot import figure




#### OBJECTS
## TILE
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
    
    
class World:
    def __init__(self, timesteps, X, Y, start_CA, CAfood, clusters, foodregenz,
                 newcenterstep,
                 g0, g1, g2, g3, g4, g5, foodtoughlow, foodtoughhigh, nutrition,
                 lowfillstart, highfillstart, foodloss, newfood, 
                 foodchance, paniclevel, enz, diffenz, germenz):
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
          
        ### FOOD GENERATION
        self.clusters = clusters
        self.newcenterstep = self.timesteps/newcenterstep
        self.centers = np.random.choice(self.X, size=(1,self.clusters))
        self.center_box = (0, self.X) # defines the box that cluster centres are allowed to be in
        self.center_dev = self.X/10 # defines the standard deviation of clusters
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
        
        ## Make CAs
        self.enz = enz
        self.diffenz = diffenz
        self.germenz = germenz
        for i in range(start_CA):
            cell = CA(self)
            self.CAs.append(cell)
        ## World Output Data
        self.data = list()
    
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
#            elif til.IND == True:
#                array[locx,locy] = self.G0 + 2
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
        ## Gotta pick the best color map
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
    
#    def addFoodold(self):
#        count = self.newfood
#        while count > 0:
#            posx = random.randint(0,(self.X-1))
#            posy = random.randint(0,(self.Y-1))
#            til = self.tiles[posx,posy]
#            if til.CA == False:
#                if til.CAfood == False:
#                    if til.IND == False:
#                        chance = random.uniform(0,1)
#                        if chance < self.foodchance:
#                            til.CAfood = True
#                            til.foodtough = random.uniform(self.ftl, self.fth)
#                            til.foodvalue = til.foodtough * self.nutrition
#                            til.ftcap = til.foodtough
#                            count = count - 1
#                        else:
#                            count = count - 1
#                if count == 0:
#                    break
#        self.updateCAFGrad()
    
    def newCenters(self):
        if self.timesteps / self.newcenterstep == self.timesteps // self.newcenterstep:
            self.centers = np.random.choice(self.X, size=(1,self.clusters))
        
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
        #### NOT SURE HOW THIS WILL CHANGE THINGS
        for multi in self.Multis:
            for cell in multi.CAs:
                CAx.append(cell.x)
                CAy.append(cell.y)
        #### SO CHECK IT OUT
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
            
#        elif (gen1[0] == 2 and gen2[0] == 1) or (gen1[0] == 1 and gen2[0] == 2):
#            return
           
    
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
                        self.tiles[loc.x][loc.y].CA = True
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
        self.updateCA()
        self.updateIND()
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
                                    check = 1
    
    def AggregateCA(self, cellA, cellB):
        newind = Multi(self)
        diffcell = self.checkDifferentiate(cellA, cellB)
        ## Problem here, if differentiated cell touches unicell
        ## sometimes turns differentiated cell into germ
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
        self.updateCA()
        for cell in self.CAs:
            cell.getNeighbors(self)
            if cell.fill < self.paniclevel:
                for neighbor in cell.neighbors:
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
                            break
                        break
                    elif neighbor.IND == True:
                        coords = self.wrapper([(neighbor.x,neighbor.y)])
                        posx = coords[0][0]
                        posy = coords[0][1]
                        for multi in self.Multis:
                            for CA in multi.CAs:
                                if CA.x == posx and CA.y == posy:
                                    if CA.fill < self.paniclevel:
                                        cell.differentiate(1,self)
                                        multi.CAs.append(cell)
                                        self.tiles[cell.x][cell.y].IND = True
                                        self.tiles[cell.x][cell.y].CA = False
                                        self.CAs.remove(cell)
                                        break
                                    break
                                break
    
    def INDAggregator(self):
        result = self.checkINDAggregate()
        while result[0][0] != []:
            coord1 = result[0]
            coord2 = result[1]
            for multi in self.Multis:
                for CA in multi.CAs:
                    if CA.x == coord1[0] and CA.y == coord1[1]:
                        multiA = multi
            for multi in self.Multis:
                for CA in multi.CAs:
                    if CA.x == coord2[0] and CA.y == coord2[1]:
                        multiB = multi
            for cell in multiB.CAs:
                multiA.CAs.append(cell)
                multiB.CAs.remove(cell)
            self.Multis.remove(multiB)
            self.updateIND()
            result = self.checkINDAggregate()
                                                
    def checkINDAggregate(self):
        cellx = []
        celly = []
        posx = []
        posy = []
        for multi in self.Multis:
            for cell in multi.CAs:
                if cell.fill < self.paniclevel:
                    for neighbor in cell.neighbors:
                        if neighbor not in multi.CAs:
                            if neighbor.IND == True:
                                coords = self.wrapper([(neighbor.x,neighbor.y)])
                                posx = coords[0][0]
                                posy = coords[0][1]
                                cellx = cell.x
                                celly = cell.y
                                break
        coord1 = [cellx,celly]
        coord2 = [posx,posy]
        print(coord1,coord2)
        return(coord1,coord2)
    
    def checkDeath(self):
        for cell in self.CAs:
            if cell.fill <= 0:
                cell.end = self.steps
                self.data.append(["cell", cell.ID, cell.start, cell.end, cell.children])
                posx = cell.x
                posy = cell.y
                self.tiles[posx][posy].CA = False
                self.CAs.remove(cell)
              #  print("A cell has died :(")
            else:
                self.CAs = self.CAs
        for multi in self.Multis:
            for cell in multi.CAs:
                if cell.fill <= 0:
                    multi.end = self.steps
                    for cell in multi.CAs:
                        posx = cell.x
                        posy = cell.y
                        self.tiles[posx][posy].IND = False
                    self.data.append(["multi", multi.ID, multi.start, multi.end, multi.CAs])
                    self.Multis.remove(multi)
                    break
                    #  print("A multi has died :(")
            else:
                self.Multis = self.Multis
                
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
        
        ### ADD FOOD
        self.addFood()
        
        ### EAT
        for multi in self.Multis:
            ## Each multi eats and update chemical gradients (in function)
            multi.multieat(self)
        for cell in self.CAs:
            ## Each cell eats and update chemical gradients (in function)
            cell.eat(self)
        self.updateCAFGrad()
        self.updateCAFTough()
        
        ### MOVE
        for multi in self.Multis:
            ## Move each multi and update chemical gradiants (in function)
           # self.updateCA()
           # self.updateIND()
            multi.movemulti(self)
            self.updateCAGrad()
        
        for cell in self.CAs:
            ## Move each cell and update chemical gradiants (in function)
            ## Note that for CAS there are two move functions to choose from,
            ## comment out one.
            ## cell.movebrain(self)
           # self.updateCA()
            #self.updateIND()
            cell.movesimple(self)
            self.updateCAGrad()
        
        ### CHECKS
        ## Check cell fills for reproduction and death & possible simulation end
        self.checkCAAggregate()
        self.checkDeath()
        #self.INDAggregator()
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
         ## CA Properties
         self.ID = len(world.CAs) + 1
         self.start = world.steps
         self.end = 0
         
         self.genome = np.zeros(5)
         self.diff = False
         self.germ = False
         self.enz = world.enz
         
         self.fill = random.uniform(world.lowfillstart, world.highfillstart)
         self.x = random.randrange(0,(world.X-1),1)
         self.y = random.randrange(0,(world.Y-1),1)
         
         self.neighbors = []
         self.children = 0
         self.spaces = [False,False,False,False,False]
         
         self.brain = np.random.uniform(-1,1,size=(4,4))
         
         
         
    def __str__(self):
        return 'Fill: %s.' % (self.fill)
    
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
    
#    def checkFill(self, world):
#        self.getNeighbors(world)
#        spaces = [False,False,False,False,False]
#        counter = 0
#        for neighbor in self.neighbors:
#            if neighbor.CA == False:
#                if neighbor.CAfood == False:
#                    if neighbor.IND == False:
#                        spaces[counter] = False
#                        counter = counter + 1
#                    else: 
#                        spaces[counter] = True
#                        counter = counter + 1
#                else:
#                    spaces[counter] = True
#                    counter = counter + 1
#            else:
#                spaces[counter] = True
#                counter = counter + 1
#        return spaces
        
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
#        
    def movesimple(self, world):
        world.updateCA()
        world.updateIND()
        self.getNeighbors(world)
        motors = []
        spaces = self.spaces
        ## Two different sets of information used as input, depending on 
        ## if the cell is well-enough fed
        ## Unfortunately, panic cells will avoid food for the sake of finding 
        ## others right now (not sure if this is realistic...)
        if self.fill > world.paniclevel:
            for tile in self.neighbors:
                motors.append(tile.CAFgrad)
        else:
            for tile in self.neighbors:
                motors.append(tile.CAgrad)
        #print(spaces)
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
        #print(move)
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
        self.ID = len(world.Multis) + 1
        self.start = world.steps
        self.end = 0
        
        self.CAs = list()
        self.size = len(self.CAs)
        
        self.tnl = list()
        self.tdl = list()
        self.tcg = [0,0,0,0]
        self.spaces = [False,False,False,False,False]
        
        self.enz = 2
        
    def __str__(self):
        return 'ID: %s. Size: %s' % (self.ID, self.size)
    
    def checkFillIND(self,n,spaces,dex):
        if n.CA == True:
            #print("FULL: CA")
            spaces[dex] = True
        else:
            if spaces[dex] == True:
                spaces[dex] = True
            else: 
                spaces[dex] == False
        if n.IND == True:
            #print("FULL: IND")
            spaces[dex] = True
        else:
            if spaces[dex] == True:
                spaces[dex] = True
            else: 
                spaces[dex] == False
        if n.CAfood == True:
            #print("FULL: CA FOOD")
            spaces[dex] = True
        else:
            if spaces[dex] == True:
                spaces[dex] = True
            else: 
                spaces[dex] == False
        return spaces
        
    
    def getNeighborsIND(self, world):
        totalneighbors = []
        totaldirections = []
        totalCAFGrads = [0,0,0,0]
        spaces = [False,False,False,False,False]
        for cell in self.CAs:
            cell.getNeighbors(world)
            neighbors = cell.neighbors
            for n in neighbors:
                for c in self.CAs:
                    if n.x == c.x and n.y == c.y:
                        #print(n)
                        neighbors.remove(n)
                        #print("REMOVED")
            for n in neighbors:
                #print(n)
                if n.x == cell.x+1 or (cell.x == world.X - 1 and n.x == 0):
                    #print("RIGHT")
                    totalneighbors.append(n)
                    totaldirections.append(0)
                    totalCAFGrads[0] = totalCAFGrads[0] + n.CAFgrad
                    spaces = self.checkFillIND(n,spaces,0)
                elif n.y == cell.y+1 or (cell.y == world.Y - 1 and n.y == 0):
                    #print("UP")
                    totalneighbors.append(n)
                    totaldirections.append(1)
                    totalCAFGrads[1] = totalCAFGrads[1] + n.CAFgrad
                    spaces = self.checkFillIND(n,spaces,1)
                elif n.x == cell.x-1 or (cell.x == 0 and n.x == world.X - 1):
                    #print("LEFT")
                    totalneighbors.append(n)
                    totaldirections.append(2)
                    totalCAFGrads[2] = totalCAFGrads[2] + n.CAFgrad
                    spaces = self.checkFillIND(n,spaces,2)
                elif n.y == cell.y-1 or (cell.y == 0 and n.y == world.Y - 1):
                    #print("DOWN")
                    totalneighbors.append(n)
                    totaldirections.append(3)
                    totalCAFGrads[3] = totalCAFGrads[3] + n.CAFgrad
                    spaces = self.checkFillIND(n,spaces,3)
                else:
                    print("ERROR")
        self.tnl = totalneighbors
        self.tdl = totaldirections
        self.tcg = totalCAFGrads
        self.spaces = spaces
            
    
#    def getNeighborsIND(self, world):
#        ## tnl = total neighbor list, gives you all the neighbors of the entire
#        ## multi body
#        tnl = []
#        ## tdl = total direct list, tells you where the neighbor is in relation
#        ## to the multi
#        tdl = []
#        spaces = [False,False,False,False,False]
#        for cell in self.CAs:
#            cell.getNeighbors(world)
#            rightcoords = [(cell.x+1,cell.y)]
#            rightcoords = world.wrapper(rightcoords)
#            upcoords = [(cell.x,cell.y+1)]
#            upcoords = world.wrapper(upcoords)
#            leftcoords = [(cell.x-1,cell.y)]
#            leftcoords = world.wrapper(leftcoords)
#            downcoords = [(cell.x,cell.y-1)]
#            downcoords = world.wrapper(downcoords)
#            for neighbor in cell.neighbors:
#                tnl.append(neighbor)
#                if neighbor.x == rightcoords[0][0] and neighbor.y == rightcoords[0][1]:
#                    tdl.append(0)
#                elif neighbor.x == upcoords[0][0] and neighbor.y == upcoords[0][1]:
#                    tdl.append(1)
#                elif neighbor.x == leftcoords[0][0] and neighbor.y == leftcoords[0][1]:
#                    tdl.append(2)
#                elif neighbor.x == downcoords[0][0] and neighbor.y == downcoords[0][1]:
#                    tdl.append(3)
#                    
#        for cell in self.CAs:
#            while world.tiles[cell.x][cell.y] in tnl:
#                dex = tnl.index(world.tiles[cell.x][cell.y])
#                tnl.remove(world.tiles[cell.x][cell.y])
#                tdl.remove(tdl[dex])
#        #print(tnl)
#        #print(tdl)
#        run = len(tnl) - 1
#        for i in range(run):
#            neighbor = tnl[i]
#            if neighbor.CA == True or neighbor.IND == True or neighbor.CAfood == True:
#                spaces[tdl[i]] = True
#        self.tnl = tnl
#        ### RIGHT[0] UP[1] LEFT[2] DOWN[3]
#        self.tdl = tdl
#        self.spaces = spaces
    
    ####################################
    ##  MOVEMENT FUNCTIONS
    ####################################
    
    
    ### NEED TO ADD A MULTI BRAIN SYSTEM WHERE THE BRAIN GROWS PERCEPTUAL 
    ### INPUTS AS CELLS ARE ADDED
    
    ### YOU'RE READING EACH NEIGHBOR ONCE, BUT WHAT IF TWO CELLS ARE TOUCHING
    ### ONE NEIGHBOR? COULD RESOLVE THIS AND THE MOVEMENT THING SIMULTANEOUSLY
    
    
    def direct(self, world):
        ## Here is where you do all the inbodied math for deciding where to go
        self.getNeighborsIND(world)
        #print(self.tnl)
        motors = self.tcg
        spaces = self.spaces
        #print(spaces)
        #print(motors)
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
        #print(move)
        return(move)
    
    
#    def direct1(self, world):
#        ## Here is where you do all the inbodied math for deciding where to go
#        self.getNeighborsIND2(world)
#        #print(self.tnl)
#        motors = [0,0,0,0]
#        spaces = self.spaces
#        #print(spaces)
#        ## Fill motors with collective food sense
#        for i in range(len(self.tnl)):
#            relloc = self.tdl[i]
#            motors[relloc] = motors[relloc] + self.tnl[i].CAFgrad
#        #print(motors)
#        ## Then do the max math (best for now)
#        goodmoves = [4]
#        maxmot = max(motors)
#        bestvalue = 0
#        while maxmot >= bestvalue:
#            if maxmot >= world.G0:
#                break
#            else:
#                move = motors.index(maxmot)
#                if spaces[move] == True:
#                    motors[move] = - 10
#                    maxmot = max(motors)
#                else:
#                    goodmoves.append(move)
#                    bestvalue = maxmot
#                    motors[move] = -10
#                    maxmot = max(motors)
#        if len(goodmoves) > 1:
#            goodmoves.remove(4)
#        move = random.choice(goodmoves)
#        #print(move)
#        return(move)
    
    ## OTHER MOVEMENT IDEAS (from Josh & Kory)
        ## Have cells emit signals to one another that play into their 
        ## chemotactic decisions
        
        ## Store the actions for each cell as decisions are made, then execute
        ## all at once (instead of moving one cell at a time: keep the current
        ## state of the cells and the state at t+1) --> should implement this
        ## up top too
        
    def movemulti(self, world):
        direct = self.direct(world)
        ## I really do not know how the coordinate system works.....
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
        #print(cell)
        world.updateIND()
        world.updateCAGrad()
        self.getNeighborsIND(world)
        for CA in self.CAs:
            if world.tiles[CA.x][CA.y].CA == True and world.tiles[CA.x][CA.y].IND == True:
                print("CHECK 2")
                
    ####################################
    ##  EATING FUNCTIONS
    ####################################
    
    ## BUSTED, DON'T GIVE IND AN ENZ, ALSO ONLY EATS ONCE PER NEIGHBOR EVEN
    ## IF TWO ADJACENT CELLS
        
    def multieat(self, world):
        for neighbor in self.tnl:
            if neighbor.CAfood == True:
                posx = neighbor.x
                posy = neighbor.y
                world.tiles[posx][posy].foodtough = world.tiles[posx][posy].foodtough - self.enz
                if world.tiles[posx][posy].foodtough <= 0:
                    size = len(self.CAs)
                    for cell in self.CAs:
                        cell.fill = cell.fill + (world.tiles[posx][posy].foodvalue/size)
                    world.tiles[posx][posy].CAfood = False
                    world.tiles[posx][posy].foodtough = 0.0
                    world.tiles[posx][posy].ftcap = 0.0
        for cell in self.CAs:
            cell.fill = cell.fill - world.foodloss
