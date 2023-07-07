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
import Lib.Blender.Core
#   ../Lib/Blender/Utilities
import Lib.Blender.Utilities
#   ../Lib/Collision_Detection/Collider/Core
import Lib.Collision_Detection.Collider.Core as Collider
#   ../Lib/Collision_Detection/Primitives
import Lib.Collision_Detection.Primitives as Primitives
#   ../Lib/Transformation/Core
from Lib.Transformation.Core import Homogeneous_Transformation_Matrix_Cls as HTM_Cls

"""
Description:
    Open Raycast.blend from the Blender folder and copy + paste this script and run it.

    Terminal:
        $ cd Documents/GitHub/Collision_Detection/Blender
        $ blender Raycast.blend
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
# Properties of the line segment:
#   Initial position of the points of the line segment a, b.
CONST_LINE_SEGMENT = [[0.0, -1.0, 0.0], [0.0, 1.0, 0.0]]
# Set the structure of the main parameters of the camera.
CONST_CAMERA_TYPE = Lib.Blender.Parameters.Camera.Right_View_Camera_Parameters_Str

def main():
    """
    Description:
        A simple script to evaluate an algorithm to check if a line segment intersects with 3D primitive object.

        The 3D primitive object in our case can be:
            1\ Axis-aligned Bounding Box (AABB)
            2\ Oriented Bounding Box (OBB)
    """

    # Deselect all objects in the current scene.
    Lib.Blender.Utilities.Deselect_All()

    # Set the camera (object) transformation and projection.
    if Lib.Blender.Utilities.Object_Exist('Camera'):
        Lib.Blender.Utilities.Set_Camera_Properties('Camera', CONST_CAMERA_TYPE)
        
    # Removes the intersections of the aligned box (AABB or OBB).
    Lib.Blender.Utilities.Remove_Object(f'Intersection_ID_0')
    Lib.Blender.Utilities.Remove_Object(f'Intersection_ID_1')

    # If the points of the line segment do not exist, create them.
    #   Note: 
    #       If the points exist, just translate/rotate them using the control panel or another method in Blender.
    for i, const_l_s in enumerate(CONST_LINE_SEGMENT):
        if Lib.Blender.Utilities.Object_Exist(f'Line_Segmet_Point_{i}') == False:
            # Properties of the created object.
            point_i_properties = {'transformation': {'Radius': 0.025, 'Location': const_l_s}, 
                                  'material': {'RGBA': [0.1,0.1,0.1,1.0], 'alpha': 1.0}}
            
            # Create a primitive three-dimensional object (sphere) with additional properties.
            Lib.Blender.Utilities.Create_Primitive('Sphere', f'Line_Segmet_Point_{i}', point_i_properties)
        
    line_segment = np.array([bpy.data.objects['Line_Segmet_Point_0'].location, 
                             bpy.data.objects['Line_Segmet_Point_1'].location], dtype=np.float32)

    # Create a class to visualize a line segment.
    LS_Poly = Lib.Blender.Core.Poly_3D_Cls('Line_Segmet_ID_0', {'bevel_depth': 0.005, 'color': [0.1,0.1,0.1,1.0]}, 
                                          {'visibility': False, 'radius': None, 'color': None})
    # Initialize the size (length) of the polyline data set.
    LS_Poly.Initialization(line_segment.shape[0])

    for i, l_s in enumerate(line_segment):           
        LS_Poly.Add(i, l_s)
       
    # Visualization of a 3-D (dimensional) polyline in the scene.
    LS_Poly.Visualization()

    # Create a specific class to work with a line segment (ray).
    Line_Segment_Cls_id_0 = Primitives.Line_Segment_Cls(line_segment[0], line_segment[1])

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
                 
    # Check if a line segment intersects with 3D primitive object (AABB or OBB). The function also contains information about 
    # where the line segment intersects the box.
    (is_intersection, points) = Box_Cls.Raycast(Line_Segment_Cls_id_0)

    if is_intersection == True:
        # Properties of the created object.
        sphere_1_properties = {'transformation': {'Radius': 0.025, 'Location': points[0]}, 
                               'material': {'RGBA': [0.0,0.0,0.0,1.0], 'alpha': 1.0}}
        sphere_2_properties = {'transformation': {'Radius': 0.025, 'Location': points[1]}, 
                               'material': {'RGBA': [0.0,0.0,0.0,1.0], 'alpha': 1.0}}
                         
        # Create a primitive three-dimensional object (sphere) with additional properties.
        Lib.Blender.Utilities.Create_Primitive('Sphere', 'Intersection_ID_0', sphere_1_properties)
        Lib.Blender.Utilities.Create_Primitive('Sphere', 'Intersection_ID_1', sphere_2_properties)

        # There is an intersection: 
        #   The color of the object is set to red.
        Lib.Blender.Utilities.Set_Object_Material_Color(CONST_BOX_NAME, [1.0,0.0,0.0,1.0])
    else:
        # There is no intersection: 
        #   The color of the object is set to green.
        Lib.Blender.Utilities.Set_Object_Material_Color(CONST_BOX_NAME, [0.0,1.0,0.0,1.0])

if __name__ == '__main__':
    main()