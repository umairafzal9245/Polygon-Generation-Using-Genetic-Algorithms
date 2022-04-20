import copy
import random
import math
import matplotlib.pyplot as plt
from shapely.geometry import LineString
from shapely.ops import polygonize, unary_union

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __eq__(self, other):
        if( self.x == other.x and self.y == other.y):
            return True
        return False
    def Print(self):
        print("x : "+str(self.x))
        print("y : "+str(self.y))

def orientation(p, q, r):

    val = (float(q.y - p.y) * (r.x - q.x)) - (float(q.x - p.x) * (r.y - q.y))
    if (val > 0):
        return 1
    elif (val < 0):
        return 2
    else:
        return 0

def doIntersect(p1, q1, p2, q2):

    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)
    if((o1 != 0) and (o3 != 0)):
        if ((o1 != o2) and (o3 != o4)):
            return True

    return False

def calculateAngle(p1, q1, p2, q2):

    BA = [p1.x - q1.x, p1.y - q1.y]
    BC = [q2.x - p2.x,q2.y - p2.y]
    dotproduct = 0
    for a, b in zip(BA, BC):
         dotproduct = dotproduct + a * b
    ABDIST = math.sqrt((BA[0] ** 2) + (BA[1] **2))
    BCDIST = math.sqrt((BC[0] ** 2) + (BC[1] **2))
    angle = math.acos(dotproduct /(ABDIST*BCDIST))
    degre = math.degrees(angle)
    if (dotproduct < 0):
        degre = abs(degre-360)

    return round(abs(degre), 4)

def calculateareaofpolygon(coordinates):

    ls = LineString(coordinates)
    lr = LineString(ls.coords[:] + ls.coords[0:1])
    mls = unary_union(lr)
    splitted = list(polygonize(mls))
    ratio = splitted[0].area
    for i in range(1,len(splitted)):
        ratio /= splitted[i].area

    return round(ratio,4)


def CreateBinary(Number):
    return bin(int(Number))[2:].zfill(8)

def CreateNumber(binary):
    return int(str(binary),2)

def generatePopulation(totalpoints,populationlength):

    totalpopulation = []
    for i in range(0,populationlength):
        chromosome = []
        for i in range(0,totalpoints):
            x = random.randint(0,255)
            y = random.randint(0,255)
            point = [CreateBinary(x),CreateBinary(y)]
            chromosome.append(point)
        totalpopulation.append(chromosome)
    return totalpopulation

def ConvertChromosometoNumber(chromosome):

    NumberChromosome = []
    for items in chromosome:
        point = [CreateNumber(items[0]),CreateNumber(items[1])]
        NumberChromosome.append(point)
    return NumberChromosome

def ConvertChromosometoBinary(chromosome):
    BinaryChromosome = []
    for items in chromosome:
        point = [CreateBinary(items[0]), CreateBinary(items[1])]
        BinaryChromosome.append(point)
    return BinaryChromosome

def PlotPolygon(coordinates):

    corr = copy.deepcopy(coordinates)
    corr.append(coordinates[0])
    xs, ys = zip(*corr)
    plt.figure()
    plt.plot(xs,ys)
    plt.show()

def pairwise(iterable):
    a = iter(iterable)
    return zip(a, a)

def GeneticAlgorithm(generations, PopulationSize,points):

    if points < 3:
        print("Invalid Points")
        return

    Population = generatePopulation(points, PopulationSize)
    gen = 0
    while gen < generations:
        popselected = selection(Population, PopulationSize)
        # for items in popselected:
        #     print(items)
        print("generation "+str(gen))
        newlist = []
        for p1,p2 in pairwise(popselected):
            o1,o2 = crossover(p1,p2)
            newlist.append(o1)
            newlist.append(o2)
            p1x = Mutation(p1)
            p2x = Mutation(p2)
            if p1x != None:
                popselected.remove(p1)
                popselected.append(p1x)
            if p2x != None:
                popselected.remove(p2)
                popselected.append(p2x)
        Population = popselected + newlist
        best, chromosome = evaluate(Population)

        if best <= 0:
            PlotPolygon(ConvertChromosometoNumber(chromosome))
            print(best)
            break
        gen = gen+1

    return


def selection(population,selectionsize):

    Chromosomewithfitness = []
    for items in population:
        fitness = FitnessFunction(ConvertChromosometoNumber(items))
        Chromosomewithfitness.append([fitness,items])
    sumoffitness = sum(w[0] for w in Chromosomewithfitness)
    ChromosomewithNormalFitness = []
    for items in Chromosomewithfitness:
        chr = [round((items[0]/sumoffitness),4),items[1]]
        ChromosomewithNormalFitness.append(chr)

    selectpopulation = []
    max = 1
    while True:
        current = 0
        rand = round(random.uniform(0,max),4)
        for item in ChromosomewithNormalFitness:
            current += item[0]
            if current >= rand:
                selectpopulation.append(item[1])
                max -= current
                ChromosomewithNormalFitness.remove(item)
                break
        if len(selectpopulation) == selectionsize:
            break

    return selectpopulation

def crossover(p1,p2):

    split1 = random.randint(0, len(p1))
    child1 = p1[0:split1] + p2[split1:]
    child2 = p2[0:split1] + p1[split1:]
    return child1,child2

def Mutation(parent):

    parentfitness = FitnessFunction(ConvertChromosometoNumber(parent))
    Mutatedchild = parent
    rand = random.randint(0,len(parent[1])-1)
    x = Mutatedchild[rand][0].replace('1','2').replace('0','1').replace('2','0')
    y = Mutatedchild[rand][1].replace('1','2').replace('0','1').replace('2','0')
    parent[rand] = [x,y]
    parentmutatedfitness = FitnessFunction(ConvertChromosometoNumber(Mutatedchild))

    if parentmutatedfitness < parentfitness:
        return None

    return Mutatedchild

def evaluate(population):

    Chromosomewithfitness = []
    for items in population:
        fitness = FitnessFunction(ConvertChromosometoNumber(items))
        Chromosomewithfitness.append([fitness, items])

    sortedlist = []
    for items in sorted(Chromosomewithfitness):
        sortedlist.append(items)

    best = sortedlist[0]

    return best[0],best[1]

def FitnessFunction(cor):

    val = 0
    totallines = []
    for i in range(0, len(cor) - 1):
        totallines.append([cor[i], cor[i + 1]])
    totallines.append([cor[len(cor) - 1], cor[0]]);
    for i in range(0,len(totallines)):
        for j in range(i+1,len(totallines)):
            p1 = Point(totallines[i][0][0], totallines[i][0][1])
            q1 = Point(totallines[i][1][0],totallines[i][1][1])
            p2 = Point(totallines[j][0][0], totallines[j][0][1])
            q2 = Point(totallines[j][1][0], totallines[j][1][1])
            if doIntersect(p1,q1,p2,q2) is True:
                val += 1
    if(val > 0):
        c = calculateareaofpolygon(cor)
        val += c
    elif (val == 0):
        totalanglelines = copy.deepcopy(totallines)
        totalanglelines.append(totalanglelines[0])
        for i in range(0, len(totalanglelines)):
            for j in range(i + 1, len(totalanglelines)):
                p1 = Point(totalanglelines[i][0][0], totalanglelines[i][0][1])
                q1 = Point(totalanglelines[i][1][0], totalanglelines[i][1][1])
                p2 = Point(totalanglelines[j][0][0], totalanglelines[j][0][1])
                q2 = Point(totalanglelines[j][1][0], totalanglelines[j][1][1])
                if (q1 == p2):
                    angle = calculateAngle(p1,q1,p2,q2)
                    if angle >= 180:
                        val += math.cos(angle)

    return val

if __name__ == '__main__':

    GeneticAlgorithm(generations=5,PopulationSize=50,points=6)