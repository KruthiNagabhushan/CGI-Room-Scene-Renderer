import math

import glm
import numpy as np

from rit_window import *
from vertex import *


class CGIengine:
    def __init__(self, myWindow, defaction):
        self.w_width = myWindow.width
        self.w_height = myWindow.height
        self.win = myWindow
        self.keypressed = 1
        self.model_transform = np.identity(4)
        self.Omatrix = np.identity(4)
        self.Pmatrix = np.identity(4)
        self.lookAtMatrix = np.identity(4)
        self.default_action = defaction
        self.normalize_matrix = np.matrix([[2 / (self.w_width - 1), 0, 0, -1],
                                           [0, 2 / (self.w_height - 1), 0, -1],
                                           [0, 0, -2 / 1000, -1],
                                           [0, 0, 0, 1]])
        self.inv = np.linalg.inv(self.normalize_matrix)
        self.depth_buffer = np.full((self.w_width, self.w_height), -float('inf'))
        self.light_position = None
        self.light_color = None
        self.ambient_color = None
        self.eye = None

        self.transform_stack = [np.identity(4)]


        # draw a line from (x0, y0) to (x1, y1) in (r,g,b)
    def rasterizeLine(self, x0, y0, x1, y1, r, g, b):
        dy = y1 - y0
        dx = x1 - x0
        if dx == 0:
            for y in range(min(y0, y1), max(y0, y1) + 1):
                self.win.set_pixel(x0, y, r, g, b)
        elif 0 <= (dy // dx) <= 1:
            if x1 < x0 or y1 < y0:
                x0, y0, x1, y1 = x1, y1, x0, y0
            self.positive_slope(x0, y0, x1, y1, r, g, b)
        elif -1 <= (dy // dx) < 0:
            self.negative_slope(x0, y0, x1, y1, r, g, b)
        elif dy // dx < -1:
            self.larger_negative_slope(x0, y0, x1, y1, r, g, b)
        elif (dy // dx) > 1:
            self.larger_slope(x0, y0, x1, y1, r, g, b)


    def positive_slope(self, x0, y0, x1, y1, r, g, b):
        dy = y1 - y0
        dx = x1 - x0
        if dy // dx <= 1:
            incE = 2 * dy
            incNE = 2 * (dy - dx)
            d = incE - dx
            y = y0
            for x in range(x0, x1 + 1):
                self.win.set_pixel(x, y, r, g, b)
                if d <= 0:
                    d += incE
                else:
                    y = y + 1
                    d += incNE
        elif dy // dx > 1:
            incE = 2 * dy
            incNE = 2 * (dy - dx)
            d = incE + dx
            y = y0
            x = x0
            self.win.set_pixel(x, y, r, g, b)
            for y in range(y0, y1 + 1):
                self.win.set_pixel(x, y, r, g, b)
                if d > 0:
                    d += incE
                else:
                    d += incNE
                    x = x + 1


    def negative_slope(self, x0, y0, x1, y1, r, g, b):
        dy = y1 - y0
        dx = x1 - x0
        if dy < 0:
            incE = 2 * dy
            incSE = 2 * (dx + dy)
            d = incE + dx
            y = y0
            x = x0
            self.win.set_pixel(x, y, r, g, b)
            for x in range(x0, x1 + 1):
                self.win.set_pixel(x, y, r, g, b)
                if d <= 0:
                    d += incSE
                    y = y - 1
                else:
                    d += incE
        elif dx < 0:
            x0, y0, x1, y1 = x1, y1, x0, y0
            dy = y1 - y0
            dx = x1 - x0
            incE = 2 * dy
            incSE = 2 * (dx + dy)
            d = incE + dx
            y = y0
            x = x0
            self.win.set_pixel(x, y, r, g, b)
            for x in range(x0, x1 + 1):
                self.win.set_pixel(x, y, r, g, b)
                if d <= 0:
                    d += incSE
                    y = y - 1
                else:
                    d += incE


    def larger_negative_slope(self, x0, y0, x1, y1, r, g, b):
        dy = y1 - y0
        dx = x1 - x0
        if dy < 0:
            x0, y0, x1, y1 = x1, y1, x0, y0
            dy = y1 - y0
            dx = x1 - x0
            incE = 2 * dx
            incSE = 2 * (dy + dx)
            d = incE + dy
            y = y0
            x = x0
            self.win.set_pixel(x, y, r, g, b)
            for y in range(y0, y1 + 1):
                self.win.set_pixel(x, y, r, g, b)
                if d > 0:
                    d += incE
                else:
                    x = x - 1
                    d += incSE

        elif dx < 0:
            incE = 2 * dx
            incSE = 2 * (dy + dx)
            d = incE + dy
            y = y0
            x = x0
            self.win.set_pixel(x, y, r, g, b)
            for y in range(y0, y1 + 1):
                self.win.set_pixel(x, y, r, g, b)
                if d > 0:
                    d += incE
                else:
                    x = x - 1
                    d += incSE


    def larger_slope(self, x0, y0, x1, y1, r, g, b):
        dy = y1 - y0
        dx = x1 - x0
        if dx > 0 and dy > 0:
            incE = 2 * dx
            incSE = 2 * (dx - dy)
            d = incE - dy
            y = y0
            x = x0
            self.win.set_pixel(x, y, r, g, b)
            for y in range(y0, y1 + 1):
                self.win.set_pixel(x, y, r, g, b)
                if d <= 0:
                    d += incE
                else:
                    x = x + 1
                    d += incSE
        else:
            x0, y0, x1, y1 = x1, y1, x0, y0
            incE = 2 * dx
            incSE = 2 * (dx - dy)
            d = incE - dy
            y = y0
            x = x0
            self.win.set_pixel(x, y, r, g, b)
            for y in range(y0, y1 + 1):
                if d >= 0:
                    x = x + 1
                    d += incE
                else:
                    d += incSE

                self.win.set_pixel(x, y, r, g, b)

    # go is called on every update of the window display loop
    # have your engine draw stuff in the window.
    def go(self):
        if self.keypressed == 1:
            # default scene
            self.default_action()
        
        if self.keypressed == 2:
            # add you own unique scene here
            self.win.clearFB (0, 0, 0)
            
        
        # push the window's framebuffer to the window
        self.win.applyFB()
        
    def keyboard (self, key) :
        if key == '1':
            self.keypressed = 1
            self.go()
        if key == '2':
            self.keypressed = 2
            self.go()



    def clearModelTransform (self):
        self.model_transform = np.identity(4)


    def translate (self, x, y, z):
        T = np.array([[1, 0,0, x],[0, 1, 0,y],[0,0, 1,z],[0,0,0,1]])
        self.model_transform= T @ self.model_transform



    def scale(self, x, y, z):
        T = np.array([[x, 0, 0,0],[0, y, 0,0],[0,0,z,0],[0, 0, 0,1]])
        self.model_transform = T @ self.model_transform


    def rotateX(self, angle):
        angle = np.radians(angle)
        T = np.matrix([[1,0,0,0],[0,math.cos(angle), math.sin(angle), 0],[0,-math.sin(angle), math.cos(angle), 0],[0,0, 0, 1]])
        self.model_transform= np.dot(T,self.model_transform)



    def rotateY(self, angle):
        angle = np.radians(angle)
        T = np.matrix(
            [[math.cos(angle), 0, -math.sin(angle), 0], [0, 1, 0, 0], [math.sin(angle), 0, math.cos(angle), 0],
             [0, 0, 0, 1]])
        self.model_transform= np.dot(T,self.model_transform)

    def rotateZ(self, angle):
        angle = np.radians(angle)
        T = np.matrix(
            [[math.cos(angle), math.sin(angle), 0, 0], [-math.sin(angle), math.cos(angle), 0, 0], [0, 0, 1, 0],
             [0, 0, 0, 1]])
        self.model_transform = np.dot(T, self.model_transform)

    def defineClipWindow(self, t, b, r, l):
        T = np.array([[2 / (r - l), 0, 0, ((-2 * l) / (r - l)) - 1],
                      [0, 2 / (t - b), 0, ((-2 * b) / (t - b)) - 1],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
        # self.transform_stack[-1] = np.dot(T,  self.transform_stack[-1])

    def defineViewWindow(self, t, b, r, l):
        T = np.array([[2 / (r - l), 0, 0, ((-2 * l) / (r - l)) - 1],
                      [0, 2 / (t - b), 0, ((-2 * b) / (t - b)) - 1],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
        self.model_transform = np.dot(T, self.model_transform)

    def pushTransform(self):
        self.transform_stack.append(self.transform_stack[-1] @ self.model_transform )
        self.model_transform = np.identity(4)

    def popTransform(self):
        self.transform_stack.pop()
        self.model_transform = np.identity(4)

    def rasterizeTriangle(self, p0, p1, p2, outr, outg, outb):
        xmin = min(p0.x, p1.x, p2.x)
        xmax = max(p0.x, p1.x, p2.x)
        ymin = min(p0.y, p1.y, p2.y)
        ymax = max(p0.y, p1.y, p2.y)

        for x in range(xmin, xmax + 1):
            for y in range(ymin, ymax + 1):
                if self.inside_test(p0, p1, p2, x, y):
                    L1, L2, L3 = self.calculate_barycentric_coordinates(p0, p1, p2, x, y)

                    z = L1 * p0.z + L2 * p1.z + L3 * p2.z
                    if z > self.depth_buffer[x, y]:  # Check against depth buffer
                        self.depth_buffer[x, y] = z
                        if (outr >= 0 or outb >= 0 or outg >= 0) and (
                                L1 <= abs(0.1) or L2 <= abs(0.1) or L3 <= abs(0.1)):
                            self.win.set_pixel(x, y, outr, outg, outb)
                        else:
                            p_r, p_g, p_b = self.calculate_interpolate_colors(p0, p1, p2, L1, L2, L3)
                            self.win.set_pixel(x, y, p_r, p_g, p_b)

    def calculate_interpolate_colors(self, p0, p1, p2, L1, L2, L3):
        p_r = int(L1 * p0.r + L2 * p1.r + L3 * p2.r)
        p_g = int(L1 * p0.g + L2 * p1.g + L3 * p2.g)
        p_b = int(L1 * p0.b + L2 * p1.b + L3 * p2.b)
        return p_r, p_g, p_b

    def calculate_barycentric_coordinates(self, p0, p1, p2, x, y):
        d = self.edge_function(p1, p2, p0.x, p0.y)
        if d != 0:
            L1 = self.edge_function(p1, p2, x, y) / d
            L2 = self.edge_function(p2, p0, x, y) / d
            L3 = self.edge_function(p0, p1, x, y) / d
        else:
            L1 = 1
            L2 = 1
            L3 = 1

        return L1, L2, L3

    def inside_test(self, p0, p1, p2, x, y):
        inside_1 = self.edge_function(p0, p1, x, y)
        inside_2 = self.edge_function(p1, p2, x, y)
        inside_3 = self.edge_function(p2, p0, x, y)

        return inside_1 >= 0 and inside_2 >= 0 and inside_3 >= 0

    def edge_function(self, p0, p1, x, y):
        return (x - p0.x) * (p1.y - p0.y) - (y - p0.y) * (p1.x - p0.x)

    def drawTriangles(self, vertex_pos, colors, indices):
        vertexes = []
        for j in range(len(indices)):
            vertexes.append(Vertex(0, 0, 0, 0, 0, 0,None))
        k = 0
        for i in indices:
            if i == 0:
                vertexes[k] = Vertex(vertex_pos[i], vertex_pos[i + 1], vertex_pos[i + 2], colors[i], colors[i + 1],
                                     colors[i + 2],None)
            else:
                i = i * 3
                vertexes[k] = Vertex(vertex_pos[i], vertex_pos[i + 1], vertex_pos[i + 2], colors[i], colors[i + 1],
                                     colors[i + 2],None)
            k += 1

        for vertex in vertexes:
            t_mat = (np.array([vertex.x, vertex.y, vertex.z, 1])).reshape(-1, 1)
            prod = self.inv @ self.Pmatrix @ self.Omatrix @ self.lookAtMatrix @ self.transform_stack[-1] @ t_mat
            prod = np.reshape(prod, (4, 1))

            vertex.x = round((prod[0, 0] / prod[3, 0]))
            vertex.y = round((prod[1, 0] / prod[3, 0]))
            vertex.z = round((prod[2, 0] / prod[3, 0]))

        m = 0
        for x in range(0, len(vertexes) // 3):
            self.rasterizeTriangle(vertexes[m], vertexes[m + 1], vertexes[m + 2])
            m = m + 3

    def drawTrianglesWireframe (self, vertex_pos, indices, r, g, b):
        vertexes = []
        for j in range(len(indices)):
            vertexes.append(Vertex(0, 0, 0, 0, 0, 0,None))
        k = 0
        for i in indices:
            if i == 0:
                vertexes[k] = Vertex(vertex_pos[i], vertex_pos[i + 1], vertex_pos[i + 2], r,g,b ,None)
            else:
                i = i * 3
                vertexes[k] = Vertex(vertex_pos[i], vertex_pos[i + 1], vertex_pos[i + 2], r,g,b,None)
            k += 1

        for vertex in vertexes:
            t_mat = (np.array([vertex.x,vertex.y,vertex.z,1])).reshape(-1, 1)

            prod = self.inv @self.Pmatrix @ self.Omatrix @ self.lookAtMatrix @ self.transform_stack[-1] @ t_mat
            prod = np.reshape(prod, (4, 1))

            vertex.x = round((prod[0,0]/prod[3,0]))
            vertex.y = round((prod[1,0]/prod[3,0]))
            vertex.z = round((prod[2,0]/ prod[3,0]))

        m = 0
        for x in range(0, len(vertexes) // 3):
            v0, v1, v2 = [vertexes[m].x,vertexes[m].y,vertexes[m].z], [vertexes[m + 1].x,vertexes[m + 1].y,vertexes[m + 1].z], [vertexes[m + 2].x,vertexes[m + 2].y,vertexes[m + 2].z]

            E1 = np.array(v1) - np.array(v0)
            E2 = np.array(v2) - np.array(v0)
            cross_prod = np.cross(E1, E2)
            z_comp = cross_prod[2]

            if z_comp > 0:
                self.rasterizeLine(v0[0], v0[1], v1[0], v1[1], r, g, b)
                self.rasterizeLine(v0[0], v0[1], v2[0], v2[1], r, g, b)
                self.rasterizeLine(v2[0], v2[1], v1[0], v1[1], r, g, b)

            m = m + 3

    def setCamera(self, eye, lookat,up):
        self.eye = eye
        n = self.normalize(eye - lookat)
        u = self.normalize(np.cross(up, n))
        v = np.cross(n, u)
        view_matrix = np.eye(4)
        view_matrix[0,:] = [u[0],u[1],u[2],-1 * np.dot(u, eye)]
        view_matrix[1,:] = [v[0],v[1],v[2],-1 * np.dot(v, eye)]
        view_matrix[2,:] = [n[0], n[1], n[2], -1 * np.dot(n, eye)]
        view_matrix[3,:] = [0,0,0,1]


        self.lookAtMatrix = view_matrix

    def normalize(self, vec):
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec
        return vec / norm

    def setOrtho(self, l, r, b, t, n, f):
        self.Omatrix = np.array([[2/(r-l),0,0,-(r+l)/(r-l)],
                   [0,2/(t-b),0,-(t+b)/(t-b)],
                   [0,0,-2/(f-n),-(f+n)/(f-n)],
                   [0, 0, 0, 1]])
    def setFrustum(self, l, r, b, t, n, f):
        self.Pmatrix = np.array([
            [(2 * n) / (r - l), 0, (r + l) / (r - l), 0],
            [0, (2 * n) / (t - b), (t + b) / (t - b), 0],
            [0, 0, -(f + n) / (f - n), (-2 * f * n) / (f - n)],
            [0, 0, -1, 0]
        ])


    def drawTrianglesC (self, vertex_pos, indices, r, g,b, outr, outg, outb):
        vertexes = []
        for j in range(len(indices)):
            vertexes.append(Vertex(0, 0, 0, 0, 0, 0, None))
        k = 0
        for i in indices:
            if i == 0:
                vertexes[k] = Vertex(vertex_pos[i], vertex_pos[i + 1], vertex_pos[i + 2],r,g,b, None)
            else:
                i = i * 3
                vertexes[k] = Vertex(vertex_pos[i], vertex_pos[i + 1], vertex_pos[i + 2], r,g,b, None)
            k += 1

        m = 0

        for vertex in vertexes:
            t_mat = (np.array([vertex.x, vertex.y, vertex.z, 1])).reshape(-1, 1)
            prod = self.inv @ self.Pmatrix @ self.Omatrix @ self.lookAtMatrix @ self.transform_stack[-1] @ t_mat
            prod = np.reshape(prod, (4, 1))

            vertex.x = round((prod[0, 0] / prod[3, 0]))
            vertex.y = round((prod[1, 0] / prod[3, 0]))
            vertex.z = round((prod[2, 0] / prod[3, 0]))


        m = 0

        for x in range(0, len(vertexes) // 3):
            self.rasterizeTriangle(vertexes[m], vertexes[m + 1], vertexes[m + 2], outr,outg,outb)
            m = m + 3


    def drawTrianglesPhong (self, vertex_pos, indices, normals, ocolor, scolor, k, exponent, doGouraud):
        vertexes = []
        for j in range(len(indices)):
            vertexes.append(Vertex(0, 0, 0, 0, 0, 0,None))
        q = 0
        for i in indices:
            if i == 0:
                ambient_value = self.calculate_ambient_value(ocolor)
                # normal_vector = np.array([self.transformNormal(normals[i]), self.transformNormal(normals[i + 1]),
                #                           self.transformNormal(normals[i + 2])])
                normal_vector = np.array([normals[i], normals[i + 1],
                                          normals[i + 2]])
                vertex_position = np.array([vertex_pos[i], vertex_pos[i + 1], vertex_pos[i + 2]])
                diffuse_value = self.calculate_diffuse_value(vertex_position, normal_vector, ocolor)
                specular_value = self.calculate_specular_value(vertex_position, normal_vector, scolor, exponent)
                vertex_data = {
                    "ambient_value": k[0] * ambient_value,
                    "diffuse_value": k[1] * diffuse_value,
                    "specular_value": k[2] * specular_value,
                    "normal_vector": normal_vector,
                }
                vertexes[q] = Vertex(vertex_pos[i], vertex_pos[i + 1], vertex_pos[i + 2], ocolor[0], ocolor[1], ocolor[2],vertex_data)
            else:
                i = i * 3
                ambient_value = self.calculate_ambient_value(ocolor)
                # normal_vector = np.array([self.transformNormal(normals[i]), self.transformNormal(normals[i + 1]),
                #                           self.transformNormal(normals[i + 2])])
                normal_vector = np.array([normals[i], normals[i + 1],
                                         normals[i + 2]])
                vertex_position = np.array([vertex_pos[i], vertex_pos[i + 1], vertex_pos[i + 2]])
                diffuse_value = self.calculate_diffuse_value(vertex_position, normal_vector, ocolor)
                specular_value = self.calculate_specular_value(vertex_position, normal_vector, scolor, exponent)
                vertex_data = {
                    "ambient_value": k[0] * ambient_value,
                    "diffuse_value": k[1] * diffuse_value,
                    "specular_value": k[2] * specular_value,
                    "normal_vector": normal_vector,
                }
                vertexes[q] = Vertex(vertex_pos[i], vertex_pos[i + 1], vertex_pos[i + 2], ocolor[0], ocolor[1],
                                     ocolor[2], vertex_data)
            q += 1

        for vertex in vertexes:
            t_mat = (np.array([vertex.x, vertex.y, vertex.z, 1])).reshape(-1, 1)
            prod = self.inv @ self.Pmatrix @ self.Omatrix @ self.lookAtMatrix @ self.transform_stack[-1] @ t_mat
            prod = np.reshape(prod, (4, 1))

            vertex.x = round((prod[0, 0] / prod[3, 0]))
            vertex.y = round((prod[1, 0] / prod[3, 0]))
            vertex.z = round((prod[2, 0] / prod[3, 0]))


        m = 0

        for x in range(0, len(vertexes) // 3):
            self.rasterizeTrianglePhong(vertexes[m], vertexes[m + 1], vertexes[m + 2], doGouraud,k)
            m = m + 3


    def interpolate_and_normalize(self,v1,v2,v3, bary):
        i_ambient = sum(
            w * v.data["ambient_value"] for w, v in zip(bary, [v1, v2, v3]))
        i_diffuse = sum(
            w * v.data["diffuse_value"] for w, v in zip(bary, [v1, v2, v3]))
        i_specular = sum(
            w * v.data["specular_value"] for w, v in zip(bary, [v1, v2, v3]))

        return i_ambient, i_diffuse, i_specular,
    def setLight(self, pos, C):
        self.light_position = np.array(pos)
        self.light_color = np.array(C)

    def setAmbient(self, C):
        self.ambient_color = np.array(C)

    def calculate_ambient_value(self, ocolor):
        return ocolor * self.ambient_color

    def calculate_diffuse_value(self, vertex_position, normal_vector, ocolor):
        light_direction = np.subtract(self.light_position, vertex_position)
        light_direction /= np.linalg.norm(light_direction)

        # Normalize normal vector
        # normal_vector_norm = normal_vector / np.linalg.norm(normal_vector)

        cos_theta = np.dot(normal_vector, light_direction)
        cos_theta = max(cos_theta, 0)


        if cos_theta < 0:
            diffuse_value = np.array([0.0, 0.0, 0.0])
        else:
            diffuse_value = self.light_color * ocolor * cos_theta
        return diffuse_value

    def calculate_specular_value(self,vertex_position, normal_vector, scolor, exponent):

        light_direction = np.subtract(np.array(self.light_position),vertex_position)
        light_direction /= np.linalg.norm(light_direction)

        view_vector = (self.eye - vertex_position)
        view_vector /= np.linalg.norm(view_vector)

        # normal_vector_norm = normal_vector / np.linalg.norm(normal_vector)

        reflection_vector = light_direction -  (2 * np.dot(light_direction, normal_vector) * normal_vector )
        reflection_vector /= -np.linalg.norm(reflection_vector)

        cos_alpha = np.dot(reflection_vector, view_vector)

        if cos_alpha < 0:
            specular_value = np.array([0.0, 0.0, 0.0])
        else:
            specular_value = self.light_color * scolor * (cos_alpha ** exponent)
        return specular_value


    def rasterizeTrianglePhong(self, p0, p1, p2,doGouraud,k):
        xmin = min(p0.x, p1.x, p2.x)
        xmax = max(p0.x, p1.x, p2.x)
        ymin = min(p0.y, p1.y, p2.y)
        ymax = max(p0.y, p1.y, p2.y)

        for x in range(xmin, xmax + 1):
            for y in range(ymin, ymax + 1):
                if self.inside_test(p0, p1, p2, x, y):
                    L1, L2, L3 = self.calculate_barycentric_coordinates(p0, p1, p2, x, y)
                    z = L1 * p0.z + L2 * p1.z + L3 * p2.z
                    if z > self.depth_buffer[x, y]:
                        self.depth_buffer[x, y] = z
                        if doGouraud:
                            p_ambient = L1 * p0.data["ambient_value"] + L2 * p1.data["ambient_value"] + L3 * p2.data[
                                "ambient_value"]
                            p_diffuse = L1 * p0.data["diffuse_value"] + L2 * p1.data["diffuse_value"] + L3 * p2.data[
                                "diffuse_value"]
                            p_specular = L1 * p0.data["specular_value"] + L2 * p1.data["specular_value"] + L3 * p2.data[
                                "specular_value"]
                        else:
                            p_ambient, p_diffuse, p_specular  = self.interpolate_and_normalize(p0, p1, p2, [L1, L2, L3])

                        final_color = p_ambient + p_diffuse + p_specular
                        self.win.set_pixel(x, y, final_color[0] * 255, final_color[1] * 255, final_color[2] * 255)


    def calculate_phong_color(self, ambient_value, diffuse_value, specular_value, k):
        intensity = k[0] * ambient_value + k[1] * diffuse_value + k[2] * specular_value
        color = np.clip(intensity, 0, 1)
        return color

    def transformNormal(self, N):
        M = self.model_transform
        N_vector = np.array([0.0, 0.0, N, 1.0])
        transformed_normal_homogeneous = np.dot(M, N_vector)
        transformed_normal = transformed_normal_homogeneous[:3]
        normalized_normal = transformed_normal / np.linalg.norm(transformed_normal)

        return  normalized_normal

    def drawTrianglesTextures(self, vertex_pos, indices, uvs, im):

        vertexes = []
        for j in range(len(indices)):
            vertexes.append(Vertex(0, 0, 0, 0, 0, 0, None))
        q = 0
        l = 0
        for i in indices:
            if i == 0:
                uv0 = uvs[l]
                uv1 = uvs[l + 1]
                data = {
                    "uv" : np.array([uv0,uv1])
                }
                vertexes[q] = Vertex(vertex_pos[i], vertex_pos[i + 1], vertex_pos[i + 2], 0,0,0, data)

            else:
                i = i * 3
                uv0 = uvs[l]
                uv1 = uvs[l + 1]
                data = {
                    "uv": np.array([uv0, uv1])
                }
                vertexes[q] = Vertex(vertex_pos[i], vertex_pos[i + 1], vertex_pos[i + 2], 0, 0, 0, data)
            l = l + 2
            q += 1

        for vertex in vertexes:
            t_mat = (np.array([vertex.x, vertex.y, vertex.z, 1])).reshape(-1, 1)
            prod = self.inv @ self.Pmatrix @ self.Omatrix @ self.lookAtMatrix @ self.transform_stack[-1] @ t_mat
            prod = np.reshape(prod, (4, 1))

            vertex.x = round((prod[0, 0] / prod[3, 0]))
            vertex.y = round((prod[1, 0] / prod[3, 0]))
            vertex.z = round((prod[2, 0] / prod[3, 0]))


        m = 0

        for x in range(0, len(vertexes) // 3):
            self.rasterizeTriangleTextures(vertexes[m], vertexes[m + 1], vertexes[m + 2],  im)
            m = m + 3


    def rasterizeTriangleTextures(self, p0, p1, p2,  im):
            # Perform triangle rasterization
            xmin = min(p0.x, p1.x, p2.x)
            xmax = max(p0.x, p1.x, p2.x)
            ymin = min(p0.y, p1.y, p2.y)
            ymax = max(p0.y, p1.y, p2.y)

            for x in range(xmin, xmax + 1):
                for y in range(ymin, ymax + 1):
                    if self.inside_test(p0, p1, p2, x, y):
                        # Calculate barycentric coordinates
                        L1, L2, L3 = self.calculate_barycentric_coordinates(p0, p1, p2, x, y)
                        z = L1 * p0.z + L2 * p1.z + L3 * p2.z
                        if z > self.depth_buffer[x, y]:

                            # Interpolate UV coordinates using barycentric coordinates
                            interpolated_u = L1 * p0.data["uv"][1] + L2 * p1.data["uv"][1] + L3 * p2.data["uv"][1]
                            interpolated_v = L1 * p0.data["uv"][0] + L2 * p1.data["uv"][0] + L3 * p2.data["uv"][0]

                            color = self.sample_texture(im, interpolated_u, interpolated_v)
                            print(x,y)

                            self.win.set_pixel(x, y, color[0], color[1], color[2])

    def sample_texture(self, im, u, v,):
        width, height = im.size
        u = max(0, min(u, 1))
        v = max(0, min(v, 1))

        u_pixel = int(u * (width - 1))
        v_pixel = int(v * (height - 1))

        return im.getpixel((u_pixel, v_pixel))

    def drawTrianglesMyTextures(self,vertex_pos, indices, uvs, checker_size):
        vertexes = []
        for j in range(len(indices)):
            vertexes.append(Vertex(0, 0, 0, 0, 0, 0, None))
        q = 0
        l = 0
        for i in indices:
            if i == 0:
                uv0 = uvs[l]
                uv1 = uvs[l + 1]
                data = {
                    "uv" : np.array([uv0,uv1])
                }
                vertexes[q] = Vertex(vertex_pos[i], vertex_pos[i + 1], vertex_pos[i + 2], 0,0,0, data)

            else:
                i = i * 3
                uv0 = uvs[l]
                uv1 = uvs[l + 1]
                data = {
                    "uv": np.array([uv0, uv1])
                }
                vertexes[q] = Vertex(vertex_pos[i], vertex_pos[i + 1], vertex_pos[i + 2], 0, 0, 0, data)
            l = l + 2
            q += 1

        for vertex in vertexes:
            t_mat = (np.array([vertex.x, vertex.y, vertex.z, 1])).reshape(-1, 1)
            prod = self.inv @ self.Pmatrix @ self.Omatrix @ self.lookAtMatrix @ self.transform_stack[-1] @ t_mat
            prod = np.reshape(prod, (4, 1))

            vertex.x = round((prod[0, 0] / prod[3, 0]))
            vertex.y = round((prod[1, 0] / prod[3, 0]))
            vertex.z = round((prod[2, 0] / prod[3, 0]))


        m = 0

        for x in range(0, len(vertexes) // 3):
            self.rasterizeTriangleMyTextures(vertexes[m], vertexes[m + 1], vertexes[m + 2],  checker_size)
            m = m + 3

    def rasterizeTriangleMyTextures(self, p0, p1, p2, param):
        # Perform triangle rasterization
        xmin = min(p0.x, p1.x, p2.x)
        xmax = max(p0.x, p1.x, p2.x)
        ymin = min(p0.y, p1.y, p2.y)
        ymax = max(p0.y, p1.y, p2.y)

        for x in range(xmin, xmax + 1):
            for y in range(ymin, ymax + 1):
                if self.inside_test(p0, p1, p2, x, y):
                    L1, L2, L3 = self.calculate_barycentric_coordinates(p0, p1, p2, x, y)
                    z = L1 * p0.z + L2 * p1.z + L3 * p2.z
                    if z > self.depth_buffer[x, y]:

                        interpolated_u = L1 * p0.data["uv"][1] + L2 * p1.data["uv"][1] + L3 * p2.data["uv"][1]
                        interpolated_v = L1 * p0.data["uv"][0] + L2 * p1.data["uv"][0] + L3 * p2.data["uv"][0]

                        # color = self.checkboard_texture( interpolated_u, interpolated_v, param)
                        # color = self.brick_texture(interpolated_u,interpolated_v,20,10,2)
                        color = self.mandelbrot_texture(interpolated_u, interpolated_v, param)
                        print(x, y)

                        self.win.set_pixel(x, y, color[0], color[1], color[2])


    def checkboard_texture(self, u, v, checker_size):

        u_checker = u / checker_size
        v_checker = v / checker_size

        color1 = (255, 0, 0)
        color2 = (0, 255, 0)

        u_mod = u_checker % 1.0
        v_mod = v_checker % 1.0

        if (u_mod < 0.5 and v_mod < 0.5) or (u_mod >= 0.5 and v_mod >= 0.5):
            return color1
        else:
            return color2


    def mandelbrot_texture(self, u, v, max_iter):
        real_part = (u - 0.5) * 3.0
        imag_part = (v - 0.5) * 2.0

        c = complex(real_part, imag_part)
        z = complex(0, 0)

        for i in range(max_iter):
            z = z * z + c
            if abs(z) > 2.0:
                normalized_iterations = i / max_iter
                return (
                    int(200 * normalized_iterations),
                    int(88 * (1 - normalized_iterations)),
                    130
                )


        return 255, 255, 255










