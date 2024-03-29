# BPY (Blender as a python) [pip3 install bpy]
import bpy
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# System (Default)
import sys
#   Add access if it is not in the system path.
if '../../' + 'src' not in sys.path:
    sys.path.append('../../' + 'src')
# Custom Lib.:
#   ../Blender/Parameters/Camera
import Blender.Parameters.Camera
#   ../Blender/Utilities
import Blender.Utilities
#   ../Collider/Core
import Collider.Core as Collider
#   ../Primitives/Core
import Primitives.Core as Primitives
#   ../Transformation/Core
from Transformation.Core import Homogeneous_Transformation_Matrix_Cls as HTM_Cls
    
"""
Description:
    Open Overlap.blend from the Blender folder and copy + paste this script and run it.

    Terminal:
        $ cd Documents/GitHub/Collision_Detection/Blender/Collision_Detection
        $ blender Overlap.blend
"""

"""
Description:
    Initialization of constants.
"""
# Properties of the Axis-aligned Bounding Boxes (AABBs) or Oriented Bounding 
# Boxes (OBBs):
#   Scale of the Boxes.
CONST_BOX_SCALES = [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]
#   Name of the Boxes.
#       Note: The name of the string is important because the keyword "AABB.." or "OBB.." is 
#             used to decide which algorithm to use.
CONST_BOX_NAMES  = ['AABB_ID_0', 'AABB_ID_1']
# Set the structure of the main parameters of the camera.
CONST_CAMERA_TYPE = Blender.Parameters.Camera.Right_View_Camera_Parameters_Str

def main():
    """
    Description:
        A program to evaluate an algorithm to check if two 3D primitives overlap (intersect) or not.

        The combination of the two primitives in our case can be:
            1\ AABB <-> AABB: 
                CONST_BOX_NAMES = ['AABB_ID_0', 'AABB_ID_1']
            2\ OBB <-> OBB  : 
                CONST_BOX_NAMES = ['OBB_ID_0', 'OBB_ID_1']
            3\ OBB <-> AABB : 
                CONST_BOX_NAMES = ['OBB_ID_0', 'AABB_ID_0'] or ['AABB_ID_0', 'OBB_ID_0']
    """

    # Deselect all objects in the current scene.
    Blender.Utilities.Deselect_All()

    # Set the camera (object) transformation and projection.
    if Blender.Utilities.Object_Exist('Camera'):
        Blender.Utilities.Set_Camera_Properties('Camera', CONST_CAMERA_TYPE)

    Box_Cls = [None, None]
    for i, (box_name_i, box_scale_i) in enumerate(zip(CONST_BOX_NAMES, CONST_BOX_SCALES)):
        # If the box object does not exist, create it.
        #   Note: 
        #       If the object exists, just translate/rotate it using the control panel or another method in Blender.
        if Blender.Utilities.Object_Exist(box_name_i) == False:
            # Properties of the created object.
            box_properties = {'transformation': {'Size': 1.0, 'Scale': box_scale_i, 'Location': [0.0, 0.0, 0.0]}, 
                              'material': {'RGBA': [0.8,0.8,0.8,1.0], 'alpha': 0.05}}
                                
            # Create a primitive three-dimensional object (Cube -> AABB) with additional properties.
            Blender.Utilities.Create_Primitive('Cube', box_name_i, box_properties)

        # Create a specific class to work with a box.
        Primitive_Cls = Primitives.Box_Cls([0.0, 0.0, 0.0], box_scale_i)

        # Create a specific class to work with an aligned box (AABB or OBB). The selection of algorithm depends on the keyword "CONST_BOX_NAME".
        Box_Cls[i] = Collider.OBB_Cls(Primitive_Cls) if 'OBB' in box_name_i else (Collider.AABB_Cls(Primitive_Cls) 
                                                                                  if 'AABB' in box_name_i else None)
        # Transform the box according to the input homogeneous transformation matrix.
        Box_Cls[i].Transformation(HTM_Cls(bpy.data.objects[box_name_i].matrix_basis, np.float64))

        # To evaluate the correct position/rotation of the box, find the vertices of the object.
        for j, verts_j in enumerate(Box_Cls[i].Vertices):
            if Blender.Utilities.Object_Exist(f'Vertex_ID_{i}_{j}') == True:
                bpy.data.objects[f'Vertex_ID_{i}_{j}'].location = verts_j

    # Check if two 3D primitives overlap (intersect) or not.
    if Box_Cls[0].Overlap(Box_Cls[1]) == True:
        # The boxes overlap:
        #   The color of the objects is set to red.
        Blender.Utilities.Set_Object_Material_Color(CONST_BOX_NAMES[0], [1.0,0.0,0.0,1.0])
        Blender.Utilities.Set_Object_Material_Color(CONST_BOX_NAMES[1], [1.0,0.0,0.0,1.0])
    else:
        # The boxes do not overlap:
        #   The color of the objects is set to green.
        Blender.Utilities.Set_Object_Material_Color(CONST_BOX_NAMES[0], [0.0,1.0,0.0,1.0])
        Blender.Utilities.Set_Object_Material_Color(CONST_BOX_NAMES[1], [0.0,1.0,0.0,1.0])
 
if __name__ == '__main__':
    main()
