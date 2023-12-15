"""
https://www.researchgate.net/publication/337638561_Mother_Tree_Optimization

Input: : NT , PT , d, Krs, Cl, and El
    NT : The population size (AFSs)
    PT : The position of the active food sources
    d: The dimension of the problem
    Krs: The number of kin recognition signals
    Cl: The number of climate change events (0 for MTO)
    El: The elimination percentage

Distribute T agents uniformly over the search space (P1, . . . , PT )
Evaluate the fitness value of T agents (S1 . . . ST )
Sort solutions in descending order based on the fitness value and store them in S
S = Sort(S1 . . . ST )
The sorted positions with the same rank of S stored in array A
A = (P1 . . . PT )
loop
    for krs = 1 to Krs do
        Use equations (6)-(11) to update the position of each agent in A
        Evaluate the fitness of the updated positions
        Sort solutions in descending order and store them in S
        Update A
    end for
    if Cl = 0 then
        BREAK;
    else
        Select the best agents in S ((1 - El) S)
        Store the best selected position in Abest
        Distort Abest (mulitply by random vector)
        Distort(Abest) = Abest * R(d)
        Remove the rest of the population (El)S
        Generate random agents equal to the the number of removed agents
        Cl = Cl - 1
    end if
end loop (Cl > 0)
S = Sort(S1 . . . ST )
Global Solution = Min(S)
return Global Solution
"""

import numpy as np
import math
import matplotlib.pyplot as plt

# from https://stackoverflow.com/questions/3173320/text-progress-bar-in-terminal-with-block-characters/13685020
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()



#dimensionality
d = 10

#rpot signal step size
rs_step_size = 0.06

#MFN signal step size
mfn_step_size = 0.006

#range of value of inputs
#list of d 2-tuples
#rastrigin
#ranges = [(-5.12,5.12)] * d
#three-hump camel
#ranges = [(-5,5)] * d
#sphere, salomon
#ranges = [(-100,100)] * d
#normalized schwefel
#ranges = [(-512, 512)] * d
#rosenbrock
#ranges = [(-100, 100)] * d
#griewank
ranges = [(-600, 600)] * d


#http://profesores.elo.utfsm.cl/~tarredondo/info/soft-comp/functions/node10.html
def sphere(x):
    return sum([i ** 2 for i in x]) - 450
def three_hump_camel(x): #only for d=2
    return 2 * x[0]**2 - 1.05 * x[0]**4 + x[0]**6 / 6 + x[0] * x[1] + x[1]**2
def salomon(x):
    return 1 - np.cos(2*np.pi * math.sqrt(sum([i ** 2 for i in x]))) + 0.1 * math.sqrt(sum([i ** 2 for i in x]))
def rastrigin (x): #not reasonably possible
    return 10*d + sum([i ** 2 - 10 * np.cos(2 * np.pi * i) for i in x])
def normalized_schwefel(x):
    return sum([-i * np.sin(math.sqrt(abs(i))) for i in x]) / d
def griewank(x):
    return 1 + sum([(i ** 2) / 4000 for i in x]) - math.prod([np.cos(x[i] / math.sqrt(i+1)) for i in range(d)])
def rosenbrock(x):
    return sum([(100 * (x[i] ** 2 - x[i+1]) ** 2) + (x[i] - 1) ** 2 for i in range(len(x) - 1)]) + 390

#fitness function
#input: list of d values
def fitness_funct(ar):
    #return sphere(ar)
    #return three_hump_camel(ar)
    #return salomon(ar)
    #return normalized_schwefel(ar)
    #return rastrigin(ar)
    return griewank(ar)
    #return rosenbrock(ar)

#population size
Nt = 20

#initialize positions
positions = [[((x[1] - x[0]) * np.random.random_sample() + x[0]) for x in ranges] for y in range(Nt)]

#calculate fitness
fitness = [fitness_funct(x) for x in positions]

agents = [[positions[x], fitness[x]] for x in range(Nt)]
agents.sort(key=lambda agents: agents[1])

#random vector
def Rd():
    return (2 * np.round(np.random.random_sample(d)) - 1) * np.random.random_sample(d)

def update(ags):
    """
    Equations 6-11:
    P1(xk+1) = P1(xk ) + δR(d) (5)
    R(d) = 2(round(rand(d, 1)) - 1) rand(d, 1) (6)
        where δ is the root signal step size and R(d) is a random
        vector that has been adopted based on preliminary experiments
    P1(xk+1) = P1(xk ) + ∆R(d) (7)
        where ∆ is the MFN signal step size. The user may tune the
        values of δ and ∆ depending on the optimization problem
    """
    ags[0][0] = list(np.add(ags[0][0], np.multiply(Rd(),rs_step_size))) #(5), (6)
    if fitness_funct(ags[0][0]) < ags[0][1]:
        ags[0][0] = list(np.add(ags[0][0], np.multiply(Rd(),mfn_step_size))) #(7)
    #(8)
    for fpct_agent in range(1, int(Nt/2 - 1)):
        old_positions = ags[fpct_agent][0]
        new_positions = []
        for p in range(d):
            new_p = old_positions[p]
            for i in range(fpct_agent-1):
                new_p = new_p + (1 / (fpct_agent - i + 1)) * (ags[i][0][p] - new_p)
            new_positions.append(new_p)
        """ 
        if fitness_funct(new_positions) > ags[fpct_agent][1]:
            new_positions = list(np.add(old_positions, np.multiply(Rd(), 0.05)))
        """
        ags[fpct_agent][0] = new_positions
    #(10)
    for fct_agent in range(int(Nt/2 - 1), int(Nt/2 + 2)):
        old_positions = ags[fct_agent][0]
        new_positions = []
        for p in range(d):
            new_p = old_positions[p]
            for i in range(int(fct_agent - (Nt/2 - 1)), int(fct_agent - 1)):
                new_p = new_p + (1 / (fct_agent - i + 1)) * (ags[i][0][p] - new_p)
            new_positions.append(new_p)
        ags[fct_agent][0] = new_positions
    #(11)
    for lpct_agent in range(int(Nt/2 + 2), Nt):
        old_positions = ags[lpct_agent][0]
        new_positions = []
        for p in range(d):
            new_p = old_positions[p]
            for i in range(int(lpct_agent - (Nt/2 - 1)), int(lpct_agent-1)):
                new_p = new_p + (1 / (lpct_agent - i + 1)) * (ags[i][0][p] - new_p)
            new_positions.append(new_p)
        ags[lpct_agent][0] = new_positions
    return ags

def plot_graph():
    #x, y = np.array(np.meshgrid(np.linspace(-512,512,1000), np.linspace(-512,512,1000)))
    x, y = np.array(np.meshgrid(np.linspace(-100,100,1000), np.linspace(-100,100,1000)))
    z = fitness_funct([x, y])
    x_min = x.ravel()[z.argmin()]
    y_min = y.ravel()[z.argmin()]
    plt.figure(figsize=(8,6))
    #plt.imshow(z, extent=[-512, 512, -512, 512], origin='lower', cmap='viridis', alpha=0.5)
    plt.imshow(z, extent=[-100, 100, -100, 100], origin='lower', cmap='viridis', alpha=0.5)
    plt.colorbar()
    plt.plot([x_min], [y_min], marker='o', markersize=5, color="white")
    #contours = plt.contour(x, y, z, 10, colors='black', alpha=0.4)
    #plt.clabel(contours, inline=True, fontsize=8, fmt="%.0f")
    plt.plot([agents[0][0][0]], [agents[0][0][1]], marker="x")
    plt.plot([x[0][0] for x in agents[1:]], [y[0][1] for y in agents[1:]], "bo")
    plt.show()    


best_agents = []
#krs: Kin Recognition Signals, aka number of iterations: default 10,000
#cl: climate change events: default 20
#el: elimination percentage: default 0.2 (20%)
def optimize(krs=10000, cl=20, el=0.2):
    global agents, best_agents
    while cl >= 0:
        for k in range(krs):
            printProgressBar(k, krs, '', 'Complete. ' + str(cl) + ' events remaining     ')
            #plot_graph()
            new_positions = [a[0] for a in update(agents)]
            new_fitness = [fitness_funct(x) for x in new_positions]
            agents = [[new_positions[x], new_fitness[x]] for x in range(Nt)]
            agents.sort(key=lambda agents: agents[1])
            """
            for a in range(len(agents)):
                print("Agent ",a, ": ", agents[a], sep="")
            print()
            """
            """
            if k % 100 == 0:
                print(agents[0])
            """
            best_agents.append(agents[0])
        if cl == 0:
            break
        else:
            #Distort Abest (mulitply by random vector)
            agents[0][0] = list(np.multiply(agents[0][0], Rd()))
            agents[0][1] = fitness_funct(agents[0][0])
            #Remove the rest of the population (El)S
            #Generate random agents equal to the the number of removed agents
            for i in range(int((1 - el) * Nt), Nt):
                new_random_pos = [((x[1] - x[0]) * np.random.random_sample() + x[0]) for x in ranges]
                agents[i] = [new_random_pos, fitness_funct(new_random_pos)]
            cl = cl - 1

optimize(cl=5)
agents.sort(key=lambda agents: agents[1])
best_agent = agents[0]



print()
#rastrigin, sphere, three-hump camel, salomon, griewank
true_min = [[0] * d, 0]
#shifted sphere
#true_min = [[0] * d, -450]
#normalized schwefel
#true_min = [[420.968746]* d, -418.9828872724337998]
#rosenbrock
#true_min = [[1] * d, 390]
difference = best_agent[1] - true_min[1]
percent_error = abs(100* difference / true_min[1])
print("Best Agent:   ", best_agent)
print("True Minimum: ", true_min)
print("Difference:   ", difference)
print("Percent Error:", percent_error)


#plot_graph()


#plt.plot([a[1] for a in best_agents[-500:]])
#plt.show()
