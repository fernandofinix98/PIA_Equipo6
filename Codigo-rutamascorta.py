# Ahora Python
from collections import deque
import math
import networkx as nx
import matplotlib.pyplot as plt
import time

class Grafo:
    def __init__(self,graph_dict=None, directed=True):
        self.vertices = []
        self.matriz = [[None]*0 for i in range(0)]
        self.graph_dict = graph_dict or {}
        self.directed = directed
        if not directed:
            self.make_undirected()
        #Atributos para dibujar el grafo
        self.dGraph = []
        self.dLabel = []
        self.nodoInicio=None
        self.nodoFin=None
    # Create an undirected graph by adding symmetric edges
    def make_undirected(self):
        for a in list(self.graph_dict.keys()):
            for (b, dist) in self.graph_dict[a].items():
                self.graph_dict.setdefault(b, {})[a] = dist
    
    def dibujarGrafo(self):
        draw_graph(self.dGraph,self.dLabel)
    
    def addConnection(self, nodo1, nodo2, label):
        connection = (nodo1, nodo2)
        self.dGraph.append(connection)
        self.dLabel.append(label)
    
    #Setters y getters
    def setHeuristicos(self, heuristicos):
        self.heuristicos = heuristicos
    def getHeuristicos(self):
        return self.heuristicos
    
    def getNodoInicio(self):
        return self.nodoInicio
    def setNodoInicio(self, nodoInicio):
        self.nodoInicio = nodoInicio
        
    def getNodoFin(self):
        return self.nodoFin
    def setNodoFin(self, nodoFin):
        self.nodoFin = nodoFin
    
    # Add a link from A and B of given distance, and also add the inverse link if the graph is undirected
    def connect(self, A, B, distance=1):
        self.graph_dict.setdefault(A, {})[B] = distance
        if not self.directed:
            self.graph_dict.setdefault(B, {})[A] = distance
    # Get neighbors or a neighbor
    def get(self, a, b=None):
        links = self.graph_dict.setdefault(a, {})
        if b is None:
            return links
        else:
            return links.get(b)
    # Return a list of nodes in the graph
    def nodes(self):
        s1 = set([k for k in self.graph_dict.keys()])
        s2 = set([k2 for v in self.graph_dict.values() for k2, v2 in v.items()])
        nodes = s1.union(s2)
        return list(nodes)
    def imprimir_matriz(self, m, texto):
        cadena = ""

        for c in range(len(m)):
            cadena += "\t" + str(self.vertices[c])

        cadena += "\n " + ("   -" * len(m))

        for f in range(len(m)):
            cadena += "\n" + str(self.vertices[f]) + " |"
            for c in range(len(m)):
                if texto:
                    cadena += "\t" + str(m[f][c])
                else:
                    if f == c and (m[f][c] is None or m[f][c] == 0):
                        cadena += "\t" + "0"
                    else:
                        if m[f][c] is None or math.isinf(m[f][c]):
                            cadena += "\t" + "X"
                        else:
                            cadena += "\t" + str(m[f][c])

        cadena += "\n"
        print(cadena)

    @staticmethod
    def contenido_en(lista, k):
        if lista.count(k) == 0:
            return False
        return True

    def esta_en_vertices(self, v):
        if self.vertices.count(v) == 0:
            return False
        return True

    def agregar_vertices(self, v):
        if self.esta_en_vertices(v):
            return False 
        self.vertices.append(v)
        filas = columnas = len(self.matriz)
        matriz_aux = [[None] * (filas+1) for i in range(columnas+1)]
        for f in range(filas):
            for c in range(columnas):
                matriz_aux[f][c] = self.matriz[f][c]
        self.matriz = matriz_aux
        return True

    def agregar_arista(self, inicio, fin, valor, dirijida):
        if not(self.esta_en_vertices(inicio)) or not(self.esta_en_vertices(fin)):
            return False
        self.matriz[self.vertices.index(inicio)][self.vertices.index(fin)] = valor
        if not dirijida:
            self.matriz[self.vertices.index(fin)][self.vertices.index(inicio)] = valor
        self.addConnection(inicio, fin, valor)
        return True

    def obtener_sucesores(self, v):
        pos_vertice = self.vertices.index(v)

        list_sucesores = []

        for i in range(len(self.matriz)):
            if self.matriz[pos_vertice][i] is not None:
                list_sucesores.append(self.vertices[i])

        return list_sucesores

    # Aciclico.
    def camino(self, k, v2):
        # Con ciclos.
        return self.__camino(k, v2, [])

    def __camino(self, k, v2, visitados):
        if k == v2:
            return True

        visitados.append(k)
        sucesores = self.obtener_sucesores(k)

        for vertice in sucesores:
            if not self.contenido_en(visitados, vertice):
                if self.__camino(vertice, v2, visitados):
                    return True

        return False

class Node:
    # Initialize the class
    def __init__(self, name:str, parent:str):
        self.name = name
        self.parent = parent
        self.g = 0 # Distance to start node
        self.h = 0 # Distance to goal node
        self.f = 0 # Total cost
    # Compare nodes
    def __eq__(self, other):
        return self.name == other.name
    # Sort nodes
    def __lt__(self, other):
         return self.f < other.f
    # Print node
    def __repr__(self):
        return ('({0},{1})'.format(self.name, self.f))
# A* search
def astar_search(graph):
    startTime = time.time()
    # Create lists for open nodes and closed nodes
    open = []
    closed = []
    # Create a start node and an goal node
    start_node = Node(g.getNodoInicio(), None)
    goal_node = Node(g.getNodoFin(), None)
    # Add the start node
    open.append(start_node)
    
    # Loop until the open list is empty
    while len(open) > 0:
        # Sort the open list to get the node with the lowest cost first
        open.sort()
        # Get the node with the lowest cost
        current_node = open.pop(0)
        # Add the current node to the closed list
        closed.append(current_node)
        
        # Check if we have reached the goal, return the path
        if current_node == goal_node:
            path = []
            while current_node != start_node:
                path.append(current_node.name)
                #path.append(current_node.name + ': ' + str(current_node.g))
                current_node = current_node.parent
            path.append(start_node.name)
            #path.append(start_node.name + ': ' + str(start_node.g))
            endTime = time.time()
            # Return reversed path
            return path[::-1], (endTime - startTime)*1000
        # Get neighbours
        neighbors = graph.get(current_node.name)
        # Loop neighbors
        for key, value in neighbors.items():
            # Create a neighbor node
            neighbor = Node(key, current_node)
            # Check if the neighbor is in the closed list
            if(neighbor in closed):
                continue
            # Calculate full path cost
            neighbor.g = current_node.g + graph.get(current_node.name, neighbor.name)
            neighbor.h = g.getHeuristicos().get(neighbor.name)
            neighbor.f = neighbor.g + neighbor.h
            # Check if neighbor is in open list and if it has a lower f value
            if(add_to_open(open, neighbor) == True):
                # Everything is green, add neighbor to open list
                open.append(neighbor)
    endTime = time.time()
    # Return None, no path is found
    return None, -1

#Breadth First Search
def bfs(graph):
        startTime = time.time()

        recorrido = []
        cola = deque([g.getNodoInicio()])

        while len(cola) > 0:
            v_aux = cola.popleft()
            recorrido.append(v_aux)
            if v_aux == g.getNodoFin():
                endTime = time.time()
                return recorrido, (endTime - startTime)*1000

            for i in range(len(graph.matriz)):
                if graph.matriz[graph.vertices.index(v_aux)][i] is not None:
                    v_candidato = graph.vertices[i]
                    if not graph.contenido_en(recorrido, v_candidato) and not graph.contenido_en(cola, v_candidato):
                        cola.append(v_candidato)

        
        return recorrido, -1
    
#Depth First Search
def dfs(graph):
        startTime = time.time()

        recorrido = []
        pila = [g.getNodoInicio()]

        while len(pila) > 0:
            v_aux = pila.pop()
            recorrido.append(v_aux)
            if v_aux == g.getNodoFin():
                endTime = time.time()
                return recorrido, (endTime - startTime)*1000

            if not graph.contenido_en(recorrido, v_aux):
                recorrido.append(v_aux)

            condicion = True

            for i in range(len(graph.matriz)):
                if graph.matriz[graph.vertices.index(v_aux)][i] is not None:
                    v_candidato = graph.vertices[i]

                    # al parecer se puede reemplazar por "and not self.contenido_en(pila, v_candidato)
                    if not graph.contenido_en(recorrido, v_candidato) and condicion:
                        # Es como un Break.
                        condicion = False

                        pila.append(v_aux)
                        pila.append(v_candidato)
        endTime = time.time()
        return recorrido, -1


# Check if a neighbor should be added to open list
def add_to_open(open, neighbor):
    for node in open:
        if (neighbor == node and neighbor.f < node.f):
            return False
    return True
   
def draw_graph(graph, labels=None, graph_layout='shell',
               node_size=1600, node_color='blue', node_alpha=0.3,
               node_text_size=12,
               edge_color='blue', edge_alpha=0.3, edge_tickness=1,
               edge_text_pos=0.3,
               text_font='sans-serif'):

    # create networkx graph
    G=nx.Graph()

    # add edges
    for edge in graph:
        G.add_edge(edge[0], edge[1])

    # these are different layouts for the network you may try
    # shell seems to work best
    if graph_layout == 'spring':
        graph_pos=nx.spring_layout(G)
    elif graph_layout == 'spectral':
        graph_pos=nx.spectral_layout(G)
    elif graph_layout == 'random':
        graph_pos=nx.random_layout(G)
    else:
        graph_pos=nx.shell_layout(G)

    # draw graph
    nx.draw_networkx_nodes(G,graph_pos,node_size=node_size, 
                           alpha=node_alpha, node_color=node_color)
    nx.draw_networkx_edges(G,graph_pos,width=edge_tickness,
                           alpha=edge_alpha,edge_color=edge_color)
    nx.draw_networkx_labels(G, graph_pos,font_size=node_text_size,
                            font_family=text_font)

    if labels is None:
        labels = range(len(graph))

    edge_labels = dict(zip(graph, labels))
    nx.draw_networkx_edge_labels(G, graph_pos, edge_labels=edge_labels, 
                                 label_pos=edge_text_pos)

    # show graph
    plt.show()

def grafo1():
    g=Grafo()
    #Agregando vertices
    g.agregar_vertices("A")
    g.agregar_vertices("B")
    g.agregar_vertices("C")
    g.agregar_vertices("D")
    g.agregar_vertices("E")
    g.agregar_vertices("F")
    g.agregar_vertices("G")
    g.agregar_vertices("H")
    g.agregar_vertices("I")
    g.agregar_vertices("J")
    g.agregar_vertices("K")
    g.agregar_vertices("L")
    g.agregar_vertices("M")
    g.agregar_vertices("N")
    g.agregar_vertices("O")
    g.agregar_vertices("P")
    g.agregar_vertices("Q")
    g.agregar_vertices("R")
    g.agregar_vertices("S")
    g.agregar_vertices("T")
    #Agregando aristas
    g.agregar_arista("A", "B", 9, True)
    g.agregar_arista("A", "C", 2, True)
    g.agregar_arista("A", "D", 4, True)
    g.agregar_arista("B", "E", 8, True)
    g.agregar_arista("C", "F", 1, True)
    g.agregar_arista("D", "I", 10, True)
    g.agregar_arista("E", "H", 2, True)
    g.agregar_arista("F", "G", 6, True)
    g.agregar_arista("I", "M", 5, True)
    g.agregar_arista("H", "K", 1, True)
    g.agregar_arista("G", "J", 7, True)
    g.agregar_arista("G", "K", 5, True)
    g.agregar_arista("G", "I", 8, True)
    g.agregar_arista("M", "O", 1, True)
    g.agregar_arista("K", "N", 6, True)
    g.agregar_arista("J", "L", 2, True)
    g.agregar_arista("L", "N", 3, True)
    g.agregar_arista("N", "P", 1, True)
    g.agregar_arista("O", "Q", 1, True)
    g.agregar_arista("P", "R", 14, True)
    g.agregar_arista("Q", "R", 7, True)
    g.agregar_arista("R", "S", 12, True)
    g.agregar_arista("P","T",10, True)
    g.agregar_arista("S","T",6,True)
    heuristicos={}
    heuristicos['A']=0
    heuristicos['B']=5
    heuristicos['C']=1
    heuristicos['D']=2
    heuristicos['E']=3
    heuristicos['F']=0
    heuristicos['G']=3
    heuristicos['H']=0
    heuristicos['I']=2
    heuristicos['J']=1
    heuristicos['K']=0
    heuristicos['L']=0
    heuristicos['M']=0
    heuristicos['N']=0
    heuristicos['O']=0
    heuristicos['P']=0
    heuristicos['Q']=1
    heuristicos['R']=6
    heuristicos['S']=5
    heuristicos['T']=4
    g.setHeuristicos(heuristicos)
    g.connect("A", "B", 9)
    g.connect("A", "C", 2)
    g.connect("A", "D", 4)
    g.connect("B", "E", 8)
    g.connect("C", "F", 1)
    g.connect("D", "I", 10)
    g.connect("E", "H", 2)
    g.connect("F", "G", 6)
    g.connect("I", "M", 5)
    g.connect("H", "K", 1)
    g.connect("G", "J", 7)
    g.connect("G", "K", 5)
    g.connect("G", "I", 8)
    g.connect("M", "O", 1)
    g.connect("K", "N", 6)
    g.connect("J", "L", 2)
    g.connect("L", "N", 3)
    g.connect("N", "P", 1)
    g.connect("O", "Q", 1)
    g.connect("P", "R", 14)
    g.connect("Q", "R", 7)
    g.connect("R", "S", 12)
    g.connect("P","T",10)
    g.connect("S","T",6)
    g.setNodoInicio("A")
    g.setNodoFin("T")
    return g

def grafo2():
    g=Grafo()
    #Agregando vertices
    g.agregar_vertices("A")
    g.agregar_vertices("B")
    g.agregar_vertices("C")
    g.agregar_vertices("D")
    g.agregar_vertices("E")
    g.agregar_vertices("F")
    g.agregar_vertices("G")
    g.agregar_vertices("H")
    g.agregar_vertices("I")

    #Agregando aristas
    g.agregar_arista("A", "B", 3, True)
    g.agregar_arista("A", "C", 2, True)
    g.agregar_arista("B", "D", 4, True)
    g.agregar_arista("C", "E", 1, True)
    g.agregar_arista("C", "F", 5, True)
    g.agregar_arista("D", "G", 3, True)
    g.agregar_arista("E", "G", 6, True)
    g.agregar_arista("F", "H", 11, True)
    g.agregar_arista("G", "I", 9, True)
    g.agregar_arista("H", "I", 8, True)
    
    heuristicos={}
    heuristicos['A']=0
    heuristicos['B']=2
    heuristicos['C']=1
    heuristicos['D']=3
    heuristicos['E']=0
    heuristicos['F']=6
    heuristicos['G']=3
    heuristicos['H']=5
    heuristicos['I']=0
    
    g.setHeuristicos(heuristicos)
    g.connect("A", "B", 3)
    g.connect("A", "C", 2)
    g.connect("B", "D", 4)
    g.connect("C", "E", 1)
    g.connect("C", "F", 5)
    g.connect("D", "G", 3)
    g.connect("E", "G", 6)
    g.connect("F", "H", 11)
    g.connect("G", "I", 9)
    g.connect("H", "I", 8)
 
    g.setNodoInicio("A")
    g.setNodoFin("I")

    return g


def grafo3():
    g=Grafo()
    #Agregando vertices
    g.agregar_vertices("A")
    g.agregar_vertices("B")
    g.agregar_vertices("C")
    g.agregar_vertices("D")
   
    #Agregando aristas
    g.agregar_arista("A", "B", 2, True)
    g.agregar_arista("A", "C", 4, True)
    g.agregar_arista("B", "D", 3, True)
    g.agregar_arista("C", "D", 5, True)
    
    heuristicos={}
    heuristicos['A']=0
    heuristicos['B']=2
    heuristicos['C']=3
    heuristicos['D']=0
    
    g.setHeuristicos(heuristicos)
    g.connect("A", "B", 2)
    g.connect("A", "C", 4)
    g.connect("B", "D", 3)
    g.connect("C", "D", 5)
    g.setNodoInicio("A")
    g.setNodoFin("D")
    return g


def grafo4():
    g=Grafo()
    #Agregando vertices
    g.agregar_vertices("A")
    g.agregar_vertices("B")
    g.agregar_vertices("C")
    g.agregar_vertices("D")
    g.agregar_vertices("E")
    g.agregar_vertices("F")
    g.agregar_vertices("G")
    g.agregar_vertices("H")

    #Agregando aristas
    g.agregar_arista("A", "B", 1, True)
    g.agregar_arista("A", "C", 6, True)
    g.agregar_arista("A", "D", 5, True)
    g.agregar_arista("B", "G", 7, True)
    g.agregar_arista("C", "G", 2, True)
    g.agregar_arista("D", "E", 4, True)
    g.agregar_arista("G", "H", 8, True)
    g.agregar_arista("E", "F", 9, True)
    g.agregar_arista("F", "H", 3, True)
   
    heuristicos={}
    heuristicos['A']=0
    heuristicos['B']=4
    heuristicos['C']=1
    heuristicos['D']=3
    heuristicos['E']=2
    heuristicos['F']=0
    heuristicos['G']=1
    heuristicos['H']=0
    
    g.setHeuristicos(heuristicos)
    g.connect("A", "B", 1)
    g.connect("A", "C", 6)
    g.connect("A", "D", 5)
    g.connect("B", "G", 7)
    g.connect("C", "G", 2)
    g.connect("D", "E", 4)
    g.connect("G", "H", 8)
    g.connect("E", "F", 9)
    g.connect("F", "H", 3)
    g.setNodoInicio("A")
    g.setNodoFin("H")
    return g



def grafo5():
    g=Grafo()
    #Agregando vertices
    g.agregar_vertices("A")
    g.agregar_vertices("B")
    g.agregar_vertices("C")
    g.agregar_vertices("D")
    g.agregar_vertices("E")
    g.agregar_vertices("F")
    g.agregar_vertices("G")
    g.agregar_vertices("H")
    g.agregar_vertices("I")
    g.agregar_vertices("J")
    g.agregar_vertices("K")
   
    #Agregando aristas
    g.agregar_arista("A", "B", 6, True)
    g.agregar_arista("A", "C", 14, True)
    g.agregar_arista("B", "D", 13, True)
    g.agregar_arista("C", "D", 5, True)
    g.agregar_arista("D", "E", 4, True)
    g.agregar_arista("D", "F", 3, True)
    g.agregar_arista("D", "G", 17, True)
    g.agregar_arista("E", "I", 8, True)
    g.agregar_arista("F", "H", 2, True)
    g.agregar_arista("H", "I", 10, True)
    g.agregar_arista("H", "J", 1, True)
    g.agregar_arista("G", "H", 11, True)
    g.agregar_arista("J", "K", 9, True)
   
    heuristicos={}
    heuristicos['A']=0
    heuristicos['B']=5
    heuristicos['C']=4
    heuristicos['D']=2
    heuristicos['E']=1
    heuristicos['F']=2
    heuristicos['G']=6
    heuristicos['H']=0
    heuristicos['I']=3
    heuristicos['J']=1
    heuristicos['K']=0

    g.setHeuristicos(heuristicos)
    g.connect("A", "B", 6)
    g.connect("A", "C", 14)
    g.connect("B", "D", 13)
    g.connect("C", "D", 5)
    g.connect("D", "E", 4)
    g.connect("D", "F", 3)
    g.connect("D", "G", 17)
    g.connect("E", "I", 8)
    g.connect("F", "H", 2)
    g.connect("H", "I", 10)
    g.connect("H", "J", 1)
    g.connect("G", "H", 11)
    g.connect("J", "K", 9)   
    g.setNodoInicio("A")
    g.setNodoFin("K")
    return g



def grafo6():
    g=Grafo()
    #Agregando vertices
    g.agregar_vertices("A")
    g.agregar_vertices("B")
    g.agregar_vertices("C")
    g.agregar_vertices("D")
    g.agregar_vertices("E")
    g.agregar_vertices("F")

    #Agregando aristas
    g.agregar_arista("B", "A", 1, True)
    g.agregar_arista("A", "E", 2, True)
    g.agregar_arista("A", "C", 6, True)
    g.agregar_arista("E", "F", 3, True)
    g.agregar_arista("C", "D", 7, True)
    g.agregar_arista("D", "E", 9, True)


    heuristicos={}
    heuristicos['A']=0
    heuristicos['B']=0
    heuristicos['C']=5
    heuristicos['D']=4
    heuristicos['E']=1
    heuristicos['F']=0
    
    g.setHeuristicos(heuristicos)
    g.connect("B", "A", 1)
    g.connect("A", "E", 2)
    g.connect("A", "C", 6)
    g.connect("E", "F", 3)
    g.connect("C", "D", 7)
    g.connect("D", "E", 9)
    g.setNodoInicio("B")
    g.setNodoFin("F")
    return g




def grafo7():
    g=Grafo()
    #Agregando vertices
    g.agregar_vertices("A")
    g.agregar_vertices("B")
    g.agregar_vertices("C")
    g.agregar_vertices("D")
    g.agregar_vertices("E")
    g.agregar_vertices("F")
    g.agregar_vertices("G")
    g.agregar_vertices("H")
    g.agregar_vertices("I")
    g.agregar_vertices("J")

    #Agregando aristas
    g.agregar_arista("A", "B", 6, True)
    g.agregar_arista("A", "C", 9, True)
    g.agregar_arista("B",  "D", 12, True)
    g.agregar_arista("C", "B", 15, True)
    g.agregar_arista("C", "D", 7, True)
    g.agregar_arista("B", "E", 8, True)
    g.agregar_arista("D", "F", 13, True)
    g.agregar_arista("E", "F", 5, True)
    g.agregar_arista("F", "I", 1, True)
    g.agregar_arista("F", "G", 10, True)
    g.agregar_arista("F", "H", 2, True)
    g.agregar_arista("I", "J", 3, True)
    g.agregar_arista("G", "J", 4, True)
    g.agregar_arista("H", "J", 11, True)

    heuristicos={}
    heuristicos['A']=0
    heuristicos['B']=6
    heuristicos['C']=5
    heuristicos['D']=2
    heuristicos['E']=3
    heuristicos['F']=0
    heuristicos['G']=2
    heuristicos['H']=2
    heuristicos['I']=4
    heuristicos['J']=0
   
    g.setHeuristicos(heuristicos)
    g.connect("A", "B", 6)
    g.connect("A", "C", 9)
    g.connect("B",  "D", 12)
    g.connect("C", "B", 15)
    g.connect("C", "D", 7)
    g.connect("B", "E", 8)
    g.connect("D", "F", 13)
    g.connect("E", "F", 5)
    g.connect("F", "I", 1)
    g.connect("F", "G", 10)
    g.connect("F", "H", 2)
    g.connect("I", "J", 3)
    g.connect("G", "J", 4)
    g.connect("H", "J", 11)
    g.setNodoInicio("A")
    g.setNodoFin("J")
    return g




def grafo8():
    g=Grafo()
    #Agregando vertices
    g.agregar_vertices("A")
    g.agregar_vertices("B")
    g.agregar_vertices("C")
    g.agregar_vertices("D")
    g.agregar_vertices("E")
    g.agregar_vertices("F")
    g.agregar_vertices("G")
    g.agregar_vertices("H")
    g.agregar_vertices("I")
    g.agregar_vertices("J")
    g.agregar_vertices("K")
    
    #Agregando aristas
    g.agregar_arista("A", "B", 6, True)
    g.agregar_arista("A", "C", 4, True)
    g.agregar_arista("B", "E", 2, True)
    g.agregar_arista("C", "E", 11, True)
    g.agregar_arista("E", "D", 2, True)
    g.agregar_arista("C", "D", 8, True)
    g.agregar_arista("E", "F", 14, True)
    g.agregar_arista("D", "F", 13, True)
    g.agregar_arista("F", "H", 7, True)
    g.agregar_arista("F", "G", 10, True)
    g.agregar_arista("H", "I", 9, True)
    g.agregar_arista("I", "J", 5, True)
    g.agregar_arista("G", "J", 1, True)
    g.agregar_arista("J", "K", 3, True)

    heuristicos={}
    heuristicos['A']=0
    heuristicos['B']=4
    heuristicos['C']=1
    heuristicos['D']=1
    heuristicos['E']=3
    heuristicos['F']=5
    heuristicos['G']=0
    heuristicos['H']=0
    heuristicos['I']=2
    heuristicos['J']=1
    heuristicos['K']=0

    g.setHeuristicos(heuristicos)
    g.connect("A", "B", 6)
    g.connect("A", "C", 4)
    g.connect("B", "E", 2)
    g.connect("C", "E", 11)
    g.connect("E", "D", 2)
    g.connect("C", "D", 8)
    g.connect("E", "F", 14)
    g.connect("D", "F", 13)
    g.connect("F", "H", 7)
    g.connect("F", "G", 10)
    g.connect("H", "I", 9)
    g.connect("I", "J", 5)
    g.connect("G", "J", 1)
    g.connect("J", "K", 3)

    g.setNodoInicio("A")
    g.setNodoFin("K")
    return g




def grafo9():
    g=Grafo()
    #Agregando vertices
    g.agregar_vertices("A")
    g.agregar_vertices("B")
    g.agregar_vertices("C")
    g.agregar_vertices("D")
  
    #Agregando aristas
    g.agregar_arista("A", "B", 10, True)
    g.agregar_arista("A", "C", 4, True)
    g.agregar_arista("C", "D", 2, True)
    g.agregar_arista("B", "D", 7, True)
   
    heuristicos={}
    heuristicos['A']=0
    heuristicos['B']=2
    heuristicos['C']=3
    heuristicos['D']=1

    g.setHeuristicos(heuristicos)
    g.connect("A", "B", 10)
    g.connect("A", "C", 4)
    g.connect("C", "D", 2)
    g.connect("B", "D", 7)
    g.setNodoInicio("A")
    g.setNodoFin("D")
    return g




def grafo10():
    g=Grafo()
    #Agregando vertices
    g.agregar_vertices("A")
    g.agregar_vertices("B")
    g.agregar_vertices("C")
    g.agregar_vertices("D")
    g.agregar_vertices("E")   
    #Agregando aristas
    g.agregar_arista("A", "B", 2, True)
    g.agregar_arista("A", "D", 6, True)
    g.agregar_arista("B", "C", 8, True)
    g.agregar_arista("D", "C", 10, True)
    g.agregar_arista("D", "E", 4, True)
   
    heuristicos={}
    heuristicos['A']=0
    heuristicos['B']=1
    heuristicos['C']=5
    heuristicos['D']=3
    heuristicos['E']=0
    
    g.setHeuristicos(heuristicos)
    g.connect("A", "B", 2)
    g.connect("A", "D", 6)
    g.connect("B", "C", 8)
    g.connect("D", "C", 10)
    g.connect("D", "E", 4)
    g.setNodoInicio("A")
    g.setNodoFin("E")
    return g

def grafo11():
    g=Grafo()
    #Agregando vertices
    g.agregar_vertices("A")
    g.agregar_vertices("B")
    g.agregar_vertices("C")
    g.agregar_vertices("D")
    g.agregar_vertices("E")
    g.agregar_vertices("F")
    g.agregar_vertices("G")
   
    #Agregando aristas
    g.agregar_arista("A", "B", 8, True)
    g.agregar_arista("A", "C", 2, True)
    g.agregar_arista("C", "F", 5, True)
    g.agregar_arista("B", "C", 1, True)
    g.agregar_arista("C", "D", 2, True)
    g.agregar_arista("F", "G", 9, True)
    g.agregar_arista("D", "E", 1, True)
    g.agregar_arista("E", "F", 6, True)
    
    heuristicos={}
    heuristicos['A']=0
    heuristicos['B']=1
    heuristicos['C']=0
    heuristicos['D']=1
    heuristicos['E']=1
    heuristicos['F']=4
    heuristicos['G']=0
   
    g.setHeuristicos(heuristicos)
    g.connect("A", "B", 8)
    g.connect("A", "C", 2)
    g.connect("C", "F", 5)
    g.connect("B", "C", 1)
    g.connect("C", "D", 2)
    g.connect("F", "G", 9)
    g.connect("D", "E", 1)
    g.connect("E", "F", 6)
    g.setNodoInicio("A")
    g.setNodoFin("G")
    return g

def grafo12():
    g=Grafo()
    #Agregando vertices
    g.agregar_vertices("A")
    g.agregar_vertices("B")
    g.agregar_vertices("C")
    g.agregar_vertices("D")
    g.agregar_vertices("E")
    g.agregar_vertices("F")
    g.agregar_vertices("G")
    g.agregar_vertices("S")
   
    #Agregando aristas
    g.agregar_arista("S", "A", 5, True)
    g.agregar_arista("S", "B", 9, True)
    g.agregar_arista("S", "D", 6, True)
    
    g.agregar_arista("A", "B", 3, True)
    
    g.agregar_arista("B", "C", 1, True)
    g.agregar_arista("B", "A", 2, True)
    
    g.agregar_arista("C", "S", 6, True)
    g.agregar_arista("C", "G", 5, True)
    g.agregar_arista("C", "F", 7, True)
    
    g.agregar_arista("D", "C", 2, True)
    g.agregar_arista("D", "E", 2, True)
    
    g.agregar_arista("F", "D", 2, True)
    
    g.agregar_arista("E", "F", 7, True)
    
    heuristicos={}
    heuristicos['A']=7
    heuristicos['B']=3
    heuristicos['C']=4
    heuristicos['D']=6
    heuristicos['E']=5
    heuristicos['F']=6
    heuristicos['G']=0
    heuristicos['S']=5
   
    g.setHeuristicos(heuristicos)
    g.connect("S", "A", 5)
    g.connect("S", "B", 9)
    g.connect("S", "D", 6)
    
    g.connect("A", "B", 3)
    
    g.connect("B", "C", 1)
    g.connect("B", "A", 2)
    
    g.connect("C", "S", 6)
    g.connect("C", "G", 5)
    g.connect("C", "F", 7)
    
    g.connect("D", "C", 2)
    g.connect("D", "E", 2)
    
    g.connect("F", "D", 2)
    
    g.connect("E", "F", 7)

    g.setNodoInicio("S")
    g.setNodoFin("G")
    return g

def grafo13():
    g=Grafo()
    #Agregando vertices
    g.agregar_vertices("S")
    g.agregar_vertices("A")
    g.agregar_vertices("B")
    g.agregar_vertices("C")
    g.agregar_vertices("D")
    g.agregar_vertices("G")
   
    #Agregando aristas
    g.agregar_arista("S", "A", 1, True)
    g.agregar_arista("S", "G", 10, True)

    g.agregar_arista("A", "B", 2, True)
    g.agregar_arista("A", "C", 1, True)
    
    g.agregar_arista("B", "D", 5, True)
    
    g.agregar_arista("C", "D", 3, True)
    g.agregar_arista("C", "G", 4, True)
    
    g.agregar_arista("D", "G", 2, True)
    
    heuristicos={}
    heuristicos['A']=3
    heuristicos['B']=4
    heuristicos['C']=2
    heuristicos['D']=6
    heuristicos['S']=5
    heuristicos['G']=0

   
    g.setHeuristicos(heuristicos)
    g.connect("S", "A", 1)
    g.connect("S", "G", 10)

    g.connect("A", "B", 2)
    g.connect("A", "C", 1)
    
    g.connect("B", "D", 5)
    
    g.connect("C", "D", 3)
    g.connect("C", "G", 4)
    
    g.connect("D", "G", 2)

    g.setNodoInicio("S")
    g.setNodoFin("G")
    return g

def grafo14():
    g=Grafo()
    #Agregando vertices
    g.agregar_vertices("A")
    g.agregar_vertices("B")
    g.agregar_vertices("C")
    g.agregar_vertices("D")
    g.agregar_vertices("E")
    g.agregar_vertices("F")
    g.agregar_vertices("G")
    g.agregar_vertices("H")
    g.agregar_vertices("I")
    g.agregar_vertices("J")
    #Agregando aristas
    g.agregar_arista("A", "B", 6, True)
    g.agregar_arista("A", "F", 3, True)

    g.agregar_arista("B", "C", 3, True)
    g.agregar_arista("B", "D", 2, True)
    
    g.agregar_arista("C", "E", 5, True)
    g.agregar_arista("D", "E", 8, True)
    
    g.agregar_arista("E", "J", 5, True)
    g.agregar_arista("E", "I", 5, True)
    
    g.agregar_arista("F", "G", 1, True)
    g.agregar_arista("F", "H", 7, True)
    
    g.agregar_arista("G", "I", 3, True)
    g.agregar_arista("H", "I", 2, True)
    
    g.agregar_arista("I", "J", 3, True)
    
    heuristicos={}
    heuristicos['A']=10
    heuristicos['B']=8
    heuristicos['C']=5
    heuristicos['D']=7
    heuristicos['E']=3
    heuristicos['F']=6
    heuristicos['G']=5
    heuristicos['H']=2
    heuristicos['I']=1
    heuristicos['J']=0
   
    g.setHeuristicos(heuristicos)
    g.connect("A", "B", 6)
    g.connect("A", "F", 3)

    g.connect("B", "C", 3)
    g.connect("B", "D", 2)
    
    g.connect("C", "E", 5)
    g.connect("D", "E", 8)
    
    g.connect("E", "J", 5)
    g.connect("E", "I", 5)
    
    g.connect("F", "G", 1)
    g.connect("F", "H", 7)
    
    g.connect("G", "I", 3)
    g.connect("H", "I", 2)
    
    g.connect("I", "J", 3)

    g.setNodoInicio("A")
    g.setNodoFin("J")
    return g

def grafo15():
    g=Grafo()
    #Agregando vertices
    g.agregar_vertices("A")
    g.agregar_vertices("B")
    g.agregar_vertices("C")
    g.agregar_vertices("S")
    g.agregar_vertices("G")
    #Agregando aristas
    g.agregar_arista("S", "A", 1, True)
    g.agregar_arista("S", "B", 2, True)

    g.agregar_arista("A", "G", 13, True)
    g.agregar_arista("B", "G", 5, True)
    g.agregar_arista("B", "C", 1, True)
    
    heuristicos={}
    heuristicos['A']=5
    heuristicos['B']=3
    heuristicos['C']=2
    heuristicos['S']=6
    heuristicos['G']=0
   
    g.setHeuristicos(heuristicos)
    g.connect("S", "A", 1)
    g.connect("S", "B", 2)

    g.connect("A", "G", 13)
    g.connect("B", "G", 5)
    g.connect("B", "C", 1)
    g.setNodoInicio("S")
    g.setNodoFin("G")
    return g

def grafo16():
    g=Grafo()
    #Agregando vertices
    g.agregar_vertices("A")
    g.agregar_vertices("B")
    g.agregar_vertices("C")
    g.agregar_vertices("S")
    g.agregar_vertices("G")
    #Agregando aristas
    g.agregar_arista("S", "A", 4, True)
    g.agregar_arista("S", "B", 3, True)

    g.agregar_arista("A", "G", 1, True)
    g.agregar_arista("B", "G", 5, True)
    g.agregar_arista("A", "C", 6, True)
    
    heuristicos={}
    heuristicos['A']=2
    heuristicos['B']=5
    heuristicos['C']=2
    heuristicos['S']=4
    heuristicos['G']=0
   
    g.setHeuristicos(heuristicos)
    g.connect("S", "A", 4)
    g.connect("S", "B", 3)

    g.connect("A", "G", 1)
    g.connect("B", "G", 5)
    g.connect("A", "C", 6)
    g.setNodoInicio("S")
    g.setNodoFin("G")
    return g

def grafo17():
    g=Grafo()
    #Agregando vertices
    g.agregar_vertices("A")
    g.agregar_vertices("B")
    g.agregar_vertices("C")
    g.agregar_vertices("D")
    g.agregar_vertices("E")
    g.agregar_vertices("F")
    g.agregar_vertices("G")
    
    #Agregando aristas
    g.agregar_arista("A", "B", 2, True)
    g.agregar_arista("A", "C", 1, True)
    g.agregar_arista("A", "G", 9, True)
    
    g.agregar_arista("C", "E", 4, True)
    g.agregar_arista("C", "D", 2, True)
    
    g.agregar_arista("B", "F", 2, True)
    g.agregar_arista("B", "D", 3, True)
    
    g.agregar_arista("G", "F", 4, True)
    g.agregar_arista("G", "D", 4, True)
    
    heuristicos={}
    heuristicos['A']=6
    heuristicos['B']=3
    heuristicos['C']=6
    heuristicos['D']=1
    heuristicos['E']=10
    heuristicos['F']=4
    heuristicos['G']=0
   
    g.setHeuristicos(heuristicos)
    g.connect("A", "B", 2)
    g.connect("A", "C", 1)
    g.connect("A", "G", 9)
    
    g.connect("C", "E", 4)
    g.connect("C", "D", 2)
    
    g.connect("B", "F", 2)
    g.connect("B", "D", 3)
    
    g.connect("G", "F", 4)
    g.connect("G", "D", 4)
    
    
    g.setNodoInicio("A")
    g.setNodoFin("G")
    return g

def grafo18():
    g=Grafo()
    #Agregando vertices
    g.agregar_vertices("A")
    g.agregar_vertices("B")
    g.agregar_vertices("C")
    g.agregar_vertices("D")
    g.agregar_vertices("S")
    g.agregar_vertices("G")
    #Agregando aristas
    g.agregar_arista("S", "A", 1, True)
    g.agregar_arista("S", "B", 2, True)

    g.agregar_arista("A", "C", 2, True)
    g.agregar_arista("B", "C", 5, True)
    g.agregar_arista("B", "D", 7, True)
    g.agregar_arista("D", "G", 2, True)
    g.agregar_arista("C", "G", 4, True)
    
    heuristicos={}
    heuristicos['A']=2
    heuristicos['B']=4
    heuristicos['C']=3
    heuristicos['D']=5
    heuristicos['S']=1
    heuristicos['G']=0
   
    g.setHeuristicos(heuristicos)
    g.connect("S", "A", 1)
    g.connect("S", "B", 2)

    g.connect("A", "C", 2)
    g.connect("B", "C", 5)
    g.connect("B", "D", 7)
    g.connect("D", "G", 2)
    g.connect("C", "G", 4)
    
    g.setNodoInicio("S")
    g.setNodoFin("G")
    return g

def grafo19():
    g=Grafo()
    #Agregando vertices
    g.agregar_vertices("A")
    g.agregar_vertices("B")
    g.agregar_vertices("C")
    g.agregar_vertices("D")
    g.agregar_vertices("E")
    g.agregar_vertices("F")
    g.agregar_vertices("S")
    g.agregar_vertices("G")
    #Agregando aristas
    g.agregar_arista("S", "A", 5, True)
    g.agregar_arista("S", "B", 10, True)
    g.agregar_arista("S", "C", 5, True)

    g.agregar_arista("A", "E", 4, True)
    
    g.agregar_arista("B", "E", 2, True)
    
    g.agregar_arista("C", "B", 4, True)
    g.agregar_arista("C", "F", 6, True)
    
    g.agregar_arista("F", "E", 1, True)
    g.agregar_arista("F", "G", 3, True)
    
    g.agregar_arista("E", "D", 4, True)
    g.agregar_arista("E", "G", 6, True)
    
    g.agregar_arista("D", "G", 4, True)
    
    heuristicos={}
    heuristicos['A']=6
    heuristicos['B']=4
    heuristicos['C']=6
    heuristicos['D']=2
    heuristicos['E']=2
    heuristicos['F']=3
    heuristicos['S']=2
    heuristicos['G']=0
   
    g.setHeuristicos(heuristicos)
    g.connect("S", "A", 5)
    g.connect("S", "B", 10)
    g.connect("S", "C", 5)

    g.connect("A", "E", 4)
    
    g.connect("B", "E", 2)
    
    g.connect("C", "B", 4)
    g.connect("C", "F", 6)
    
    g.connect("F", "E", 1)
    g.connect("F", "G", 3)
    
    g.connect("E", "D", 4)
    g.connect("E", "G", 6)
    
    g.connect("D", "G", 4)
    
    g.setNodoInicio("S")
    g.setNodoFin("G")
    return g

def grafo20():
    g=Grafo()
    #Agregando vertices
    g.agregar_vertices("A")
    g.agregar_vertices("B")
    g.agregar_vertices("C")
    g.agregar_vertices("D")
    g.agregar_vertices("E")
    g.agregar_vertices("F")
    #Agregando aristas
    g.agregar_arista("A", "B", 2, True)
    g.agregar_arista("A", "E", 3, True)
    
    g.agregar_arista("B", "C", 1, True)
    g.agregar_arista("B", "F", 9, True)
    
    g.agregar_arista("E", "D", 6, True)
    
    g.agregar_arista("D", "F", 1, True)

    
    heuristicos={}
    heuristicos['A']=11
    heuristicos['B']=6
    heuristicos['C']=99
    heuristicos['D']=1
    heuristicos['E']=7
    heuristicos['F']=0
   
    g.setHeuristicos(heuristicos)
    g.connect("A", "B", 2)
    g.connect("A", "E", 3)
    
    g.connect("B", "C", 1)
    g.connect("B", "F", 9)
    
    g.connect("E", "D", 6)
    
    g.connect("D", "F", 1)
    
    g.setNodoInicio("A")
    g.setNodoFin("F")
    return g

#Los algoritmos que calculan el tiempo promedio de duración de cada búsqueda.
def tiemposAstar(g):
    tiempos = []
    n = 100
    while n > 0:
        resultado, tiempo = astar_search(g)
        tiempos.append(tiempo)
        n = n - 1
    prom = 0
    while n < 100:
        prom = prom + tiempos[n]
        n = n + 1
    prom = prom / 100
    return prom
    
def tiemposAnch(g):
    tiempos = []
    n = 100
    while n > 0:
        resultado, tiempo = bfs(g)
        tiempos.append(tiempo)
        n = n - 1
    prom = 0
    while n < 100:
        prom = prom + tiempos[n]
        n = n + 1
    prom = prom / 100
    return prom
    
def tiemposProf(g):
    tiempos = []
    n = 100
    while n > 0:
        resultado, tiempo = dfs(g)
        tiempos.append(tiempo)
        n = n - 1
    prom = 0
    while n < 100:
        prom = prom + tiempos[n]
        n = n + 1
    prom = prom / 100
    return prom

#Función de costo del algoritmo
def  costo(recorrido, heuristicos):
    i = 0
    costo = 0
    while i < len(recorrido):
        costo = costo + heuristicos[recorrido[i]]
        i = i + 1
    return costo

#Funciones que se usarán en el main.
def elegirGrafo():
    nGrafo = input("\nElija el número de grafo que desea (1-20): ")
    if nGrafo == '1':
        g = grafo1()
    elif nGrafo == '2':
        g = grafo2()
    elif nGrafo == '3':
        g = grafo3()
    elif nGrafo == '4':
        g = grafo4()
    elif nGrafo == '5':
        g = grafo5()
    elif nGrafo == '6':
        g = grafo6()
    elif nGrafo == '7':
        g = grafo7()
    elif nGrafo == '8':
        g = grafo8()
    elif nGrafo == '9':
        g = grafo9()
    elif nGrafo == '10':
        g = grafo10()
    elif nGrafo == '11':
        g = grafo11()
    elif nGrafo == '12':
        g = grafo12()
    elif nGrafo == '13':
        g = grafo13()
    elif nGrafo == '14':
        g = grafo14()
    elif nGrafo == '15':
        g = grafo15()
    elif nGrafo == '16':
        g = grafo16()
    elif nGrafo == '17':
        g = grafo17()
    elif nGrafo == '18':
        g = grafo18()
    elif nGrafo == '19':
        g = grafo19()
    elif nGrafo == '20':
        g = grafo20()
    else:
        g = None
        nGrafo = '0'
    print("--------------------------------------")
    if g == None:
        print("¡Valor fuera del rango!")
    else:
        print("Se escogió el grafo " + nGrafo + ".")
    return g, nGrafo
    
def imprimirGrafo(g):
    g.dibujarGrafo()
    
def recorrerGrafo(g):
    op = '0'
    while op != '4':
        print("--------------------------------------")
        print("\n(Elija el algoritmo que desea usar.)")
        print("1 - Breadth First Search")
        print("2 - Depth First Search")
        print("3 - A Star")
        print("4 - Volver")
        op = input("\nSu opción: ")
        tiempo = 0
        if op == '1':
            print("\n[RECORRIDO CON BFS]")
            while tiempo == 0:
                resultado, tiempo = bfs(g)
        elif op == '2':
            print("\n[RECORRIDO CON DFS]")
            while tiempo == 0:
                resultado, tiempo = dfs(g)
        elif op == '3':
            print("\n[RECORRIDO CON A*]")
            while tiempo == 0:
                resultado, tiempo = astar_search(g)
        elif op == '4':
            print()
        else:
            print("¡Opción inválida!")
        
        if op == '1' or op == '2' or op == '3':
            sRecorrido = resultado[0]
            costos = costo(resultado, g.heuristicos)
            i = 1
            while i < len(resultado):
                sRecorrido = sRecorrido + " -> " + resultado[i]
                i = i + 1
            print(sRecorrido)
            print("\nNodos recorridos: " + str(len(resultado)))
            print("Costo: " + str(costos))
            print("Tiempo: " + str(tiempo) + " microsegundos")
            print("\n(NOTA): El tiempo del algoritmo varía mucho en cada ejecución.")
            
        
    
def compAlgoritmos(g):
    nodosVisitados = []
    #Se obtiene el recorrido de los algoritmos
    recorridoBFS, t = bfs(g)
    recorridoDFS, t = dfs(g)
    recorridoAstar, t = astar_search(g)
    nodosVisitados.append(len(recorridoBFS))
    nodosVisitados.append(len(recorridoDFS))
    nodosVisitados.append(len(recorridoAstar))
    #Se calculan los tiempos promedio de los algoritmos
    tA = tiemposAstar(g)
    tB = tiemposAnch(g)
    tD = tiemposProf(g)
    
    #Creando la gráfica que compara los tiempos promedio de ejecución
    fig = plt.figure()
    ax=fig.add_subplot(111)
    algoritmos=["Breadth First", "Depth First", "A Star"]
    datos = [tB, tD, tA]
    xx=range(1, len(datos)+1)
    ax.bar(xx, datos, width=0.5, color=(0, 1, 0))
    ax.set_xticks(xx)
    ax.set_xticklabels(algoritmos)
    ax.set_ylabel("Duración promedio de los algoritmos (microsegundos)")
    plt.show()
    
    #Creando la gráfica  que compara los recorridos
    fig = plt.figure()
    ax=fig.add_subplot(111)
    xx=range(1, len(nodosVisitados)+1)
    ax.bar(xx, nodosVisitados, width=0.5, color=(1, 0, 0))
    ax.set_xticks(xx)
    ax.set_xticklabels(algoritmos)
    ax.set_ylabel("Cantidad de nodos recorridos")
    plt.show()
    
    #Creando la gráfica que compara los costos
    costoAstar = costo(recorridoAstar, g.heuristicos)
    costoDFS = costo(recorridoDFS, g.heuristicos)
    costoBFS = costo(recorridoBFS, g.heuristicos)
    
    costos = [costoBFS, costoDFS, costoAstar]
    
    fig = plt.figure()
    ax=fig.add_subplot(111)
    xx=range(1, len(costos)+1)
    ax.bar(xx, costos, width=0.5, color=(0, 0, 1))
    ax.set_xticks(xx)
    ax.set_xticklabels(algoritmos)
    ax.set_ylabel("Costos de cada búsqueda")
    plt.show()

if __name__ == "__main__":
    op = '0'
    g = Grafo()
    nGrafo = '0'
    while op != '5':
        print("--------------------------------------")
        print("╦═╗╦ ╦╔╦╗╔═╗  ╔╦╗╔═╗╔═╗  ╔═╗╔═╗╦═╗╔╦╗╔═╗")
        print("╠╦╝║ ║ ║ ╠═╣  ║║║╠═╣╚═╗  ║  ║ ║╠╦╝ ║ ╠═╣")
        print("╩╚═╚═╝ ╩ ╩ ╩  ╩ ╩╩ ╩╚═╝  ╚═╝╚═╝╩╚═ ╩ ╩ ╩\n")
        print("Opciones:")
        if nGrafo != '0':
            print("1 - Elegir grafo (Actual: " + nGrafo + ")")
        else:
            print("1 - Elegir grafo (Actual: Ninguno)")
        print("2 - Imprimir grafo")
        print("3 - Recorrer grafo")
        print("4 - Calcular tiempo de ejecución de grafos")
        print("5 - Salir")
        op = input("Elija una opción: ")
        #Elegir grafo
        if op == '1':
            g, nGrafo = elegirGrafo()
            if nGrafo != '0':
                inicio = None
                fin = None
        #Dibujar grafo
        elif op == '2':
            if nGrafo != '0':
                imprimirGrafo(g)
            else:
                print("--------------------------------------")
                print("¡Elija primero un grafo!")
        #Búsquedas dentro del grafo
        elif op == '3':
            if nGrafo != '0':
                recorrerGrafo(g)
            else:
                print("--------------------------------------")
                print("¡Elija primero un grafo!")
        #Gráficas para comparar resultados de los grafos
        elif op == '4':
            if nGrafo != '0':
                compAlgoritmos(g)
            else:
                print("--------------------------------------")
                print("¡Elija primero un grafo!")
        #Salir
        elif op=='5':
            print("--------------------------------------")
            print("¡Gracias por usar el programa!")
        else:
            print("--------------------------------------")
            print("Opción inválida")