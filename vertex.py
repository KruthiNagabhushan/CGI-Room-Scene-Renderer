

class Vertex:
    def __init__(self, x, y, z, r, g, b,data):
        self.x = x
        self.y = y
        self.z = z
        self.r = r
        self.g = g
        self.b = b
        self.data =  data if data is not None else {}
        
def getTriangleMin (p0, p1, p2):
    V = Vertex (p0.x, p0.y, p0.z,0, 0, 0)
    if p1.x < V.x:
        V.x = p1.x
    if p2.x < V.x:
        V.x = p2.x
    if p1.y < V.y:
        V.y = p1.y
    if p2.y < V.y:
        V.y = p2.y
    
    return V
    

def getTriangleMax (p0, p1, p2):
    V = Vertex (p0.x, p0.y, p0.z,0, 0, 0)
    if (p1.x > V.x):
        V.x = p1.x
    if (p2.x > V.x):
        V.x = p2.x
    if (p1.y > V.y):
        V.y = p1.y
    if (p2.y > V.y):
        V.y = p2.y
    
    return V
    
