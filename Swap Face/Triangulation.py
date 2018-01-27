import numpy as np
from scipy.spatial import Delaunay
import pdb

def getAllMarks(mark_pts, minX, maxX, minY, maxY):
    pts_lst = []
    pts_lst.append([0, 0])
    pts_lst.append([0, maxY-minY])
    pts_lst.append([maxX-minX, 0])
    pts_lst.append([maxX-minX, maxY-minY])

    pts_lst.append([0, -minY / 5.0 + maxY / 5.0])
    pts_lst.append([0, -minY * 2 / 5.0 + maxY * 2 / 5.0])
    pts_lst.append([0, -minY * 3 / 5.0 + maxY * 3 / 5.0])
    pts_lst.append([0, -minY * 4 / 5.0 + maxY * 4 / 5.0])
    pts_lst.append([maxX - minX, -minY / 5.0 + maxY / 5.0])
    pts_lst.append([maxX - minX, -minY * 2 / 5.0 + maxY * 2 / 5.0])
    pts_lst.append([maxX - minX, -minY * 3 / 5.0 + maxY * 3 / 5.0])
    pts_lst.append([maxX - minX, -minY * 4 / 5.0 + maxY * 4 / 5.0])

    pts_lst.append([-minX / 2.0 + maxX / 2.0, 0])
    pts_lst.append([-minX / 2.0 + maxX / 2.0, maxY-minY])

    pts_lst = np.asarray(pts_lst)
    mark_pts = np.concatenate((mark_pts, pts_lst), 0)
    return mark_pts

def triangulation(intermidiate_mark_pts):
    tri = Delaunay(intermidiate_mark_pts)
    return tri