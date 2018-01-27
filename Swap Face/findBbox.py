import pdb

def findBbox(mark_pts, extend_X, extend_Y_Up, extend_Y_Down, h, w):
    maxX = -100000
    maxY = -100000
    minX = 100000
    minY = 100000
    for i in range(len(mark_pts)):
        if mark_pts[i][0] > maxX:
            maxX = mark_pts[i][0]
        if mark_pts[i][1] > maxY:
            maxY = mark_pts[i][1]
        if mark_pts[i][0] < minX:
            minX = mark_pts[i][0]
        if mark_pts[i][1] < minY:
            minY = mark_pts[i][1]
    maxY = int(round(min(maxY + extend_Y_Down, h)))
    minY = int(round(max(minY - extend_Y_Up, 0)))
    maxX = int(round(min(maxX + extend_X, w)))
    minX = int(round(max(minX - extend_X, 0)))

    return minX, maxX, minY, maxY
