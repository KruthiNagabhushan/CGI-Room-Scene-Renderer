from rit_window import *
from cgI_engine import *
from vertex import *
from clipper import *
from shapes_new import *
import numpy as np
from PIL import Image


def default_action ():
   #create your scene here

   im = Image.open("Wood.jpg")

   myEngine.setLight([-2.5, -2.2, 1.5], [1.0, 1.0, 1.0])
   myEngine.setAmbient([1,1,1])



   # brick walls
   myEngine.setLight([0, -1.5, 0], [1.0, 1.0, 1.0])
   myEngine.setAmbient([0.2,0.2,0.2])
   myEngine.setCamera(np.array([0.0, 0.0, 2.0]), np.array([0, 0, 20]), np.array([0, 1, 0]))
   myEngine.setOrtho(-3.0, 3.0, -3.0, 3.0, -3.0, 3.0)
   myEngine.setFrustum(-2.0, 2.0, -2.0, 2.0, 1.0, 10.0)
   myEngine.pushTransform()
   myEngine.rotateX(90)
   myEngine.translate(0, 0, 0.0)
   myEngine.scale(5,5, 1.5)
   myEngine.pushTransform()
   myEngine.drawTrianglesTextures(cube_new, cube_new_idx, cube_new_uv, im )
   # myEngine.drawTrianglesPhong(cube_new, cube_new_idx,cube_new_normals, [1.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.2, 0.4, 0.4],
   #                             10.0, True)

   myEngine.popTransform()
   # myEngine.popTransform()
   myEngine.clearModelTransform()

   # bed
   myEngine.setCamera(np.array([-2.5, 2.0, 3.0]), np.array([-2.5, 2.0, 2.9]), np.array([0, 1, 0]))
   myEngine.translate(-1.5, 11, 0)
   myEngine.scale(3,0.5,1.5)
   myEngine.pushTransform()
   # myEngine.drawTrianglesTextures(cube_new, cube_new_idx, cube_new_uv, im1)
   myEngine.drawTrianglesPhong(cube_new, cube_new_idx, cube_new_normals, [0.5, 0.3, 0.5], [1.0, 1.0, 1.0],
                               [0.2, 0.4, 0.4], 10.0, True)

   myEngine.popTransform()
   # myEngine.popTransform()
   myEngine.clearModelTransform()

   # pillow
   myEngine.pushTransform()
   myEngine.translate(-1.8, 3, 0)
   myEngine.scale(2.25, 1.5, 0.2)
   myEngine.pushTransform()
   # myEngine.drawTrianglesTextures(cube_new, cube_new_idx, cube_new_uv, im1)
   myEngine.drawTrianglesPhong(cube_new, cube_new_idx, cube_new_normals, [0.5, 0.3, 0.5], [1.0, 1.0, 1.0],
                               [0.2, 0.4, 0.4], 10.0, True)

   myEngine.popTransform()
   myEngine.popTransform()


   # blanket
   myEngine.setCamera(np.array([-2.5, 4.0, 7.0]), np.array([-2.5, 2.0, 2.9]), np.array([0, 1, 0]))
   im2 = Image.open("Fabric.jpg")
   myEngine.pushTransform()
   myEngine.translate(-1.7, 10, 4)
   myEngine.scale(3.1, 0.68, 0.5)
   myEngine.pushTransform()
   myEngine.drawTrianglesTextures(cube_new, cube_new_idx, cube_new_uv, im2)
   # myEngine.drawTrianglesPhong(cube_new, cube_new_idx, cube_new_normals, [0.5, 0.3, 0.5], [1.0, 1.0, 1.0],
   #                             [0.2, 0.4, 0.4], 10.0, True)

   myEngine.popTransform()
   myEngine.popTransform()
   myEngine.clearModelTransform()



   # table
   myEngine.setCamera(np.array([-2.5, 2.0, 3.0]), np.array([-2.5, 2.0, 2.9]), np.array([0, 1, 0]))
   myEngine.setLight([0, -1.5, 0], [1.0, 1.0, 1.0])
   myEngine.setAmbient([0.2, 0.2, 0.2])
   myEngine.translate(-1.2,4.5,-0.5)
   myEngine.pushTransform()
   myEngine.scale(2.8, 0.2, 0.5)
   myEngine.pushTransform()
   # myEngine.drawTrianglesC(cube_new, cube_new_idx,0,100,100,0,0,0)
   myEngine.drawTrianglesPhong(cube_new,cube_new_idx,cube_new_normals,[0.0, 1.0, 1.0], [1.0, 1.0, 1.0], [0.2, 0.4, 0.4], 10.0, True)
   myEngine.popTransform()
   myEngine.popTransform()


   myEngine.pushTransform()
   myEngine.translate(-1.0,5.7,-1)
   myEngine.pushTransform()
   myEngine.scale(0.2,1,0)
   myEngine.pushTransform()
   # myEngine.drawTrianglesC(cylinder_new, cylinder_new_idx, 0,0,100,0,0,0)
   myEngine.drawTrianglesPhong(cylinder_new,cylinder_new_idx,cylinder_new_normals,[0.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.2, 0.4, 0.4], 10.0, True)
   myEngine.popTransform()
   myEngine.popTransform()
   myEngine.popTransform()
   myEngine.popTransform()
   # myEngine.popTransform()
   myEngine.pushTransform()
   myEngine.translate(-1.4, 5.2, 0)
   myEngine.pushTransform()
   myEngine.scale(0.5,0.1,0.1)
   myEngine.pushTransform()
   myEngine.drawTrianglesPhong(cube_new, cube_new_idx, cube_new_normals, [0.0, 1.0, 1.0], [1.0, 1.0, 1.0],
                               [0.2, 0.4, 0.4], 10.0, True)
   myEngine.popTransform()
   myEngine.popTransform()
   myEngine.popTransform()
   myEngine.clearModelTransform()


#    window1
   myEngine.setCamera(np.array([-2.5, 2.0, 3.0]), np.array([-2.5, 2.0, 2.9]), np.array([0, 1, 0]))
   # myEngine.setOrtho(-2.0, 2.0, -2.0, 2.0, -2.0, 2.0)
   myEngine.setLight([0, -1.5, 0], [1.0, 1.0, 1.0])
   myEngine.setAmbient([1,1,1])
   myEngine.translate(-2.4,1, 0)
   myEngine.pushTransform()
   myEngine.scale(1.5,2.5,0)
   myEngine.pushTransform()
   # myEngine.drawTrianglesC(cube_new, cube_new_idx,0,100,100,0,0,0)
   myEngine.drawTrianglesPhong(cube_new,cube_new_idx,cube_new_normals,[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.2, 0.4, 0.4], 10.0, True)
   myEngine.popTransform()
   myEngine.popTransform()
   # border
   myEngine.setLight([0, -1.5, 0], [1.0, 1.0, 1.0])
   myEngine.setAmbient([1, 1, 1])
   myEngine.translate(-2.4, 1, 0)
   myEngine.pushTransform()
   myEngine.scale(1.8,2.8, 0)
   myEngine.pushTransform()
   # myEngine.drawTrianglesC(cube_new, cube_new_idx,0,100,100,0,0,0)[0.4, 0.133, 0.0]
   myEngine.drawTrianglesPhong(cube_new, cube_new_idx, cube_new_normals, [0.4, 0.133, 0.0], [1.0, 1.0, 1.0],
                               [0.2, 0.4, 0.4], 10.0, True)
   myEngine.popTransform()
   myEngine.popTransform()


   # window door
   # myEngine.setOrtho(-2.0, 2.0, -2.0, 2.0, -2.0, 2.0)
   myEngine.setLight([0, -1.5, 0], [1.0, 1.0, 1.0])
   myEngine.setAmbient([1, 1, 1])
   myEngine.translate(-2, 0.95, -0.5)
   myEngine.pushTransform()
   myEngine.scale(1.0,2.6, 0)
   myEngine.rotateY(49)
   myEngine.pushTransform()
   # myEngine.drawTrianglesC(cube_new, cube_new_idx,0,100,100,0,0,0)
   myEngine.drawTrianglesPhong(cube_new, cube_new_idx, cube_new_normals, [0.4, 0.133, 0.0], [1.0, 1.0, 1.0],
                               [0.2, 0.4, 0.4], 10.0, True)
   myEngine.popTransform()
   myEngine.popTransform()

   myEngine.setLight([0, -1.5, 0], [1.0, 1.0, 1.0])
   myEngine.setAmbient([1, 1, 1])
   myEngine.translate(-2.75, 0.95, -0.5)
   myEngine.pushTransform()
   myEngine.scale(1.0, 2.6, 0)
   myEngine.rotateY(-49)
   myEngine.pushTransform()
   # myEngine.drawTrianglesC(cube_new, cube_new_idx,0,100,100,0,0,0)
   myEngine.drawTrianglesPhong(cube_new, cube_new_idx, cube_new_normals, [0.4, 0.133, 0.0], [1.0, 1.0, 1.0],
                               [0.2, 0.4, 0.4], 10.0, True)
   myEngine.popTransform()
   myEngine.popTransform()


   # light cone
   myEngine.setLight([0,-1.5,0], [1.0, 1.0, 1.0])
   myEngine.setAmbient([0.2, 0.2, 0.2])

   myEngine.translate(-2.5,-3,0)
   myEngine.pushTransform()
   myEngine.scale(1.8,0.8,0)
   myEngine.rotateX(210)
   myEngine.pushTransform()
   # myEngine.drawTrianglesC(cube_new, cube_new_idx,0,100,100,0,0,0)
   myEngine.drawTrianglesPhong(cone_new, cone_new_idx, cone_new_normals, [0.4, 0.133, 0.0], [1.0, 1.0, 1.0],
                               [0.2, 0.4, 0.4], 1000.0, True)
   myEngine.popTransform()
   myEngine.popTransform()
   # bulb
   myEngine.setLight([-2.5, -2.2, 1.5],[1.0, 1.0, 1.0])
   myEngine.setAmbient([0.2,0.2,0.2])
   myEngine.translate(-2.5, -2.5, 0)
   myEngine.pushTransform()
   myEngine.scale(0.8,0.8, 0)
   myEngine.pushTransform()
   # myEngine.drawTrianglesPhong(sphere_new, sphere_new_idx, sphere_new_normals, [1.0,1.0, 0.0], [1.0, 1.0, 1.0],
   #                             [0.2, 0.4, 0.4], 10.0, True)
   myEngine.drawTrianglesC(sphere_new,sphere_new_idx,255,255,0,-1,-1,-1)
   myEngine.popTransform()
   myEngine.popTransform()


   # mirror
   # border
   myEngine.setCamera(np.array([-2.5, 2.0, 3.0]), np.array([-2.5, 2.0, 2.9]), np.array([0, 1, 0]))
   myEngine.setLight([-2.5, -2.2, 1.5],[1.0, 1.0, 1.0])
   myEngine.setAmbient([0.2,0.2,0.2])
   myEngine.translate(-0.5,1, 0.5)
   myEngine.pushTransform()
   myEngine.scale(1,1.2, 0.1)
   myEngine.pushTransform()
   myEngine.drawTrianglesPhong(cube_new, cube_new_idx, cube_new_normals, [0.0,0.0, 0.0], [1.0, 1.0, 1.0],
                               [0.2, 0.4, 0.4], 10.0, True)
   myEngine.popTransform()
   myEngine.popTransform()

   im3 = Image.open("mirror.jpg")
   myEngine.setCamera(np.array([-2.5, 2.0, 3.1]), np.array([-2.5, 2.0, 2.9]), np.array([0, 1, 0]))
   myEngine.translate(0, 0.7, 0)
   myEngine.pushTransform()
   myEngine.scale(0.8, 1.0, 0)
   myEngine.pushTransform()
   myEngine.drawTrianglesTextures(cube_new, cube_new_idx, cube_new_uv, im3)
   myEngine.popTransform()
   myEngine.popTransform()
   myEngine.clearModelTransform()


   # pot
   myEngine.setCamera(np.array([-2.5, 2.0, 6]), np.array([-2.5, 2.0, 2.9]), np.array([0, 1, 0]))
   myEngine.translate(-0.2,5,1)
   myEngine.pushTransform()
   myEngine.scale(1.3,0.8, 0.1)
   myEngine.rotateZ(-90)
   myEngine.setLight([-2.5, -2.2, 1.5], [1.0, 1.0, 1.0])
   myEngine.setAmbient([0.2, 0.2, 0.2])
   myEngine.pushTransform()

   myEngine.drawTrianglesPhong(sphere_new, sphere_new_idx, sphere_new_normals, [1, 0, 0], [1.0, 1.0, 1.0],
                               [0.2, 0.4, 0.4], 1.0, True)
   myEngine.popTransform()
   myEngine.popTransform()


   # frame
   myEngine.setCamera(np.array([-2.4, 2.0, 6]), np.array([-2.5, 2.0, 2.9]), np.array([0, 1, 0]))
   myEngine.setLight([-2.5, -2.2, 1.5], [1.0, 1.0, 1.0])
   myEngine.setAmbient([0.2, 0.2, 0.2])
   myEngine.translate(-5.5,0,3.5)
   myEngine.pushTransform()
   myEngine.scale(0.5, 0.9, 0.2)
   myEngine.rotateY(90)
   myEngine.pushTransform()
   myEngine.drawTrianglesPhong(cube_new, cube_new_idx, cube_new_normals, [1, 0, 0], [1.0, 1.0, 1.0],
                               [0.2, 0.4, 0.4], 1.0, True)
   myEngine.popTransform()
   myEngine.popTransform()
   myEngine.clearModelTransform()

   # hanger
   myEngine.clearModelTransform()
   myEngine.setCamera(np.array([-2.5, 2.0, 6]), np.array([-2.5, 2.0, 2.9]), np.array([0, 1, 0]))
   myEngine.setLight([-2.5, -2.2, 1.5], [1.0, 1.0, 1.0])
   myEngine.setAmbient([0.2, 0.2, 0.2])
   myEngine.translate(-1, 1, 4.9)
   myEngine.pushTransform()
   myEngine.scale(0.5,0.1, 0.1)
   myEngine.rotateY(90)
   myEngine.pushTransform()
   myEngine.drawTrianglesPhong(cube_new, cube_new_idx, cube_new_normals, [1, 0.43, 0], [1.0, 1.0, 1.0],
                               [0.2, 0.4, 0.4], 1.0, True)
   myEngine.popTransform()
   myEngine.popTransform()










window = RitWindow(800, 800)
myEngine = CGIengine (window, default_action)

def main():
    window.run (myEngine)
    



if __name__ == "__main__":
    main()
