import math

import numpy as np
from vertex import Vertex


def calculate_outcode(P, top, bottom, right, left):
    outcode = np.array([0, 0, 0, 0])
    if P.y > top:
        outcode[0] = 1
    elif P.y < bottom:
        outcode[1] = 1
    if P.x > right:
        outcode[2] = 1
    elif P.x < left:
        outcode[3] = 1
    return outcode


def calculate_trivial_accept(outcode_p0, outcode_p1):
    re = np.array_equal(np.bitwise_or(outcode_p0 ,outcode_p1), np.array([0, 0, 0, 0]))
    return re


def calculate_trivial_reject(outcode_p0, outcode_p1):
    return not np.array_equal(np.bitwise_and(outcode_p0 ,outcode_p1), np.array([0, 0, 0, 0]))


def clip_against_left_edge(P0, P1, left):
    if P1.x - P0.x != 0:
        slope = (P1.y - P0.y)/(P1.x - P0.x)
        y = P0.y + (slope * (left - P0.x))
        return left,y
    else: return left, None


def clip_against_right_edge(P0, P1, right):
    if P1.x - P0.x != 0:
        slope = (P1.y - P0.y)/(P1.x - P0.x)
        y = P0.y + (slope * (right - P0.x))
        return right, y
    else:
        return right, None


def clip_against_bottom_edge(P0, P1, bottom):
    if P1.x - P0.x != 0:
        slope = (P1.y - P0.y) / (P1.x - P0.x)
        x = P0.x + ((bottom - P0.y)/slope)
        return x, bottom
    else:
        return P0.x, bottom


def clip_against_top_edge(P0, P1, top):
    if P1.x - P0.x != 0:
        slope = (P1.y - P0.y) / (P1.x - P0.x)
        x = P0.x + ((top - P0.y) / slope)
        return x, top
    else:
        return P0.x, top


def clipLine (P0, P1, top, bottom, right, left):

    outcode_p0 = calculate_outcode(P0, top, bottom, right, left)
    outcode_p1 = calculate_outcode(P1, top, bottom, right, left)


    while 1:
        if calculate_trivial_accept(outcode_p0, outcode_p1):
            return np.array([P0,P1])
        elif calculate_trivial_reject(outcode_p0, outcode_p1):
            return np.array([])
        else:
            if not np.array_equal(outcode_p0, np.array([0, 0, 0, 0])):
                if P0.x < left:
                    P0.x, P0.y = clip_against_left_edge(P0, P1, left)
                elif P0.x > right:
                    P0.x, P0.y = clip_against_right_edge(P0, P1, right)
                elif P0.y < bottom:
                    P0.x, P0.y = clip_against_bottom_edge(P0, P1, bottom)
                elif P0.y > top:
                    P0.x, P0.y = clip_against_top_edge(P0, P1, top)
                outcode_p0 = calculate_outcode(P0, top, bottom, right, left)
            else:
                if P1.x < left:
                    P1.x, P1.y = clip_against_left_edge(P1, P0, left)
                elif P1.x > right:
                    P1.x, P1.y = clip_against_right_edge(P1, P0, right)
                elif P1.y < bottom:
                    P1.x, P1.y = clip_against_bottom_edge(P1, P0, bottom)
                elif P1.y > top:
                    P1.x, P1.y = clip_against_top_edge(P1, P0, top)
                outcode_p1 = calculate_outcode(P1, top, bottom, right, left)


def inside_clip_plane(p, edge, side):
    if side == "right":
        if p.x > edge:
            return False
    elif side == "left":
        if p.x < edge:
            return False
    elif side == "top":
        if p.y > edge:
            return False
    elif side == "bottom":
        if p.y < edge:
            return False
    return True


def compute_intersection(s, p, edge, side):
    new_point = Vertex(0,0,0,0,0)
    if side == "right" or side == "left":
        if p.x - s.x != 0:
            slope = (p.y - s.y) / (p.x - s.x)
            y = s.y + (slope * (edge - s.x))
            new_point.x = edge
            new_point.y = y
            new_point.r = p.r
            new_point.g = p.g
            new_point.b = p.b
        else:
            new_point.x = edge
            new_point.y = None
            new_point.r = p.r
            new_point.g = p.g
            new_point.b = p.b

    else:
        if p.x - s.x != 0:
            slope = (p.y - s.y) / (p.x - s.x)
            x = s.x + ((edge - s.y) / slope)
            new_point.x = x
            new_point.y = edge
            new_point.r = p.r
            new_point.g = p.g
            new_point.b = p.b

        else:
            new_point.x = s.x
            new_point.y = edge
            new_point.r = p.r
            new_point.g = p.g
            new_point.b = p.b

    return new_point


def linear_interpolation(s, p, u, np):
    np.r = (p.r - s.r) * u + s.r
    np.g = (p.g - s.g) * u + s.g
    np.b = (p.b - s.b) * u + s.b
    return np


def calculate_u(s, p, np):
    u = math.sqrt(math.pow((np.x - s.x),2) + math.pow((np.y - s.y),2))/math.sqrt(math.pow((p.x - s.x),2) + math.pow((p.y - s.y),2))
    return u

def SHPC(edge,vertices, side):
    new_vertices = []
    length = len(vertices)
    if length!= 0:
        s = vertices[length-1]

        for p in vertices:
            if inside_clip_plane(p, edge,side):
                if not inside_clip_plane(s, edge, side):
                    new_point = compute_intersection(s,p,edge,side)
                    u = calculate_u(s, p , new_point)
                    new_point = linear_interpolation(s,p,u, new_point)
                    if new_point is not None:
                        new_vertices.append(new_point)
                new_vertices.append(p)
            else:
                if inside_clip_plane(s,edge,side):
                    new_point = compute_intersection(s, p, edge, side)
                    u = calculate_u(s, p, new_point)
                    new_point = linear_interpolation(s, p, u, new_point)
                    if new_point is not None:
                        new_vertices.append(new_point)
            s = p
    return new_vertices

def clipPoly (vertices, top, bottom, right, left):
    vertices = SHPC(right,vertices, "right")
    vertices = SHPC(top,vertices, "top")
    vertices = SHPC(left,vertices, "left")
    vertices = SHPC(bottom,vertices, "bottom")
    return np.array(vertices)


