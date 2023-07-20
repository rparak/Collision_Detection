# BPY (Blender as a python) [pip3 install bpy]
import bpy
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# System (Default)
import sys
#   Add access if it is not in the system path.
if '../' + 'src' not in sys.path:
    sys.path.append('../' + 'src')
# Custom Script:
#   ../Lib/Blender/Parameters/Camera
import Lib.Blender.Parameters.Camera
#   ../Lib/Blender/Core
import Lib.Blender.Utilities
#   ../Lib/Geometry/Collider/Core
import Lib.Geometry.Collider.Core as Collider
#   ../Lib/Geometry/Primitives
import Lib.Geometry.Primitives as Primitives
#   ../Lib/Transformation/Core
from Lib.Transformation.Core import Homogeneous_Transformation_Matrix_Cls as HTM_Cls
    
"""
Description:
    Open Point_Inside.blend from the Blender folder and copy + paste this script and run it.

    Terminal:
        $ cd Documents/GitHub/Collision_Detection/Blender
        $ blender Point_Inside.blend
"""

"""
Description:
    Initialization of constants.
"""
# Properties of the Axis-aligned Bounding Box (AABB) or Oriented Bounding 
# Box (OBB):
#   Scale of the Box.
CONST_BOX_SCALE = [0.5, 0.5, 0.5]
#   Name of the Box.
#       Note: The name of the string is important because the keyword "AABB.." or "OBB.." is 
#             used to decide which algorithm to use.
CONST_BOX_NAME  = 'AABB_ID_0'
# Properties of the generated points:
#   Position offset for random point generation.
CONST_OFFSET_RANDOM_POINTS = 0.25
#   Number of points to be generated.
CONST_NUM_OF_RANDOM_POINTS = 100
# Set the structure of the main parameters of the camera.
CONST_CAMERA_TYPE = Lib.Blender.Parameters.Camera.Right_View_Camera_Parameters_Str

def main():
    """
    Description:
        A program to evaluate an algorithm to determine if a given point is inside a geometric object.

        The geometric object in our case can be:
            1\ Axis-aligned Bounding Box (AABB)
            2\ Oriented Bounding Box (OBB)
    """

    # Deselect all objects in the current scene.
    Lib.Blender.Utilities.Deselect_All()
    
    # Set the camera (object) transformation and projection.
    if Lib.Blender.Utilities.Object_Exist('Camera'):
        Lib.Blender.Utilities.Set_Camera_Properties('Camera', CONST_CAMERA_TYPE)

    # Removes random points if they exist in the current scene.
    i = 0
    while True:
        if Lib.Blender.Utilities.Object_Exist(f'Point_ID_{i}') == True:
            Lib.Blender.Utilities.Remove_Object(f'Point_ID_{i}')
        else:
            break     
        i += 1

    # If the box object does not exist, create it.
    #   Note: 
    #       If the object exists, just translate/rotate it using the control panel or another method in Blender.
    if Lib.Blender.Utilities.Object_Exist(CONST_BOX_NAME) == False:
        # Properties of the created object (box).
        box_properties = {'transformation': {'Size': 1.0, 'Scale': CONST_BOX_SCALE, 'Location': [0.0,0.0,0.0]}, 
                          'material': {'RGBA': [0.8,0.8,0.8,1.0], 'alpha': 0.05}}
                            
        # Create a primitive three-dimensional object (Cube -> Box) with additional properties.
        Lib.Blender.Utilities.Create_Primitive('Cube', CONST_BOX_NAME, box_properties)

    # Create a specific class to work with a box.
    Primitive_Cls = Primitives.Box_Cls([0.0,0.0,0.0], CONST_BOX_SCALE)

    # Create a specific class to work with an aligned box (AABB or OBB). The selection of algorithm depends on the keyword "CONST_BOX_NAME".
    Box_Cls = Collider.OBB_Cls(Primitive_Cls) if 'OBB' in CONST_BOX_NAME else (Collider.AABB_Cls(Primitive_Cls) 
                                                                               if 'AABB' in CONST_BOX_NAME else None)
    # Transform the box according to the input homogeneous transformation matrix.
    Box_Cls.Transformation(HTM_Cls(bpy.data.objects[CONST_BOX_NAME].matrix_basis, np.float32))

    # To evaluate the correct position/rotation of the box, find the vertices of the object.
    for i, verts_i in enumerate(Box_Cls.Vertices):
        bpy.data.objects[f'Vertex_ID_0_{i}'].location = verts_i
        
    # reate a specific class to work with a point.
    Point_Cls_id_0 = Primitives.Point_Cls([0.0,0.0,0.0])

    # Generate random points in the scene with additional dependencies.
    rnd_p = np.zeros(Primitives.CONST_DIMENSION, dtype = np.float32)
    for i in range(CONST_NUM_OF_RANDOM_POINTS):
        for j, (box_size_i, box_p_i) in enumerate(zip(Box_Cls.Size, Box_Cls.T.p)):
            rnd_p[j] = np.random.uniform((-1)*(np.abs(box_size_i/2.0) + CONST_OFFSET_RANDOM_POINTS), 
                                         np.abs(box_size_i/2.0) + CONST_OFFSET_RANDOM_POINTS) + box_p_i
            
        # Transformation of point position in X, Y, Z axes.
        Point_Cls_id_0.Transformation(rnd_p)

        # Determine if a given point is located inside a geometric object.
        #   True : The Vector<float> of RGBA parameters will be set to red.
        #   False: The Vector<float> of RGBA parameters will be set to green.
        p_rgba = [1.0,0.0,0.0,1.0] if Box_Cls.Is_Point_Inside(Point_Cls_id_0) else [0.0,1.0,0.0,1.0]

        # Properties of the created object.
        sphere_1_properties = {'transformation': {'Radius': 0.01, 'Location': rnd_p}, 
                               'material': {'RGBA': p_rgba, 'alpha': 1.0}}
                 
        # Create a primitive three-dimensional object (sphere) with additional properties.
        Lib.Blender.Utilities.Create_Primitive('Sphere', f'Point_ID_{i}', sphere_1_properties)    

if __name__ == '__main__':
    main()
