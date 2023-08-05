# An Open-Source Collision Detection Library Useful for Robotics Applications

<p align="center">
<img src=https://github.com/rparak/Collision_Detection/blob/main/images/Collision_Detection_Background.png width="800" height="350">
</p>

## Requirements

**Programming Language**

```bash
Python
```

**Import Libraries**
```bash
More information can be found in the individual scripts (.py).
```

**Supported on the following operating systems**
```bash
Windows, Linux, macOS
```

## Project Description
The library can be used within the Robot Operating System (ROS), Blender, PyBullet, Nvidia Isaac, or any program that allows Python as a programming language.

A detailed description of each algorithm can be found in the library.
## 3D Shape Intersections

A description of how to run a program to evaluate an algorithm to check whether two 3D primitives overlap (intersect) or not.
1. Open Overlap.blend from the Blender folder.
2. Copy and paste the script from the evaluation folder (../overlap_boxes.py).
3. Run it and evaluate the results.
   
```bash
$ /> cd Documents/GitHub/Collision_Detection/Blender/Collision_Detection
$ ../Collision_Detection/Blender> blender Overlap.blend
```

<p align="center">
<img src=https://github.com/rparak/Collision_Detection/blob/main/images/3D_Shape_Intersections.png width="800" height="350">
</p>

```py 
# System (Default)
import sys
# Custom Script:
#   ../Lib/Blender/Utilities
import Lib.Blender.Utilities
#   ../Lib/Collider/Core
import Lib.Collider.Core as Collider
#   ../Lib/Primitives/Core
import Lib.Primitives.Core as Primitives
#   ../Lib/Transformation/Core
from Lib.Transformation.Core import Homogeneous_Transformation_Matrix_Cls as HTM_Cls
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

def main():
    """
    Description:
        A program to evaluate an algorithm to check if a line segment intersects with 3D primitive object.

        The 3D primitive object in our case can be:
            1\ Axis-aligned Bounding Box (AABB)
            2\ Oriented Bounding Box (OBB)
    """

    Box_Cls = [None, None]
    for i, (box_name_i, box_scale_i) in enumerate(zip(CONST_BOX_NAMES, CONST_BOX_SCALES)):
        # Create a specific class to work with a box.
        Primitive_Cls = Primitives.Box_Cls([0.0, 0.0, 0.0], box_scale_i)

        # Create a specific class to work with an aligned box (AABB or OBB). The selection of algorithm depends on the keyword "CONST_BOX_NAME".
        Box_Cls[i] = Collider.OBB_Cls(Primitive_Cls) if 'OBB' in box_name_i else (Collider.AABB_Cls(Primitive_Cls) 
                                                                                  if 'AABB' in box_name_i else None)
        # Transform the box according to the input homogeneous transformation matrix.
        Box_Cls[i].Transformation(HTM_Cls(bpy.data.objects[box_name_i].matrix_basis, np.float32))

    # Check if two 3D primitives overlap (intersect) or not.
    if Box_Cls[0].Overlap(Box_Cls[1]) == True:
        # The boxes overlap.
        print(True)
    else:
        # The boxes do not overlap.
        print(False)

if __name__ == '__main__':
    sys.exit(main())
```

## 3D Point Tests

A description of how to run a program to evaluate an algorithm to determine if a given point is inside a geometric object.
1. Open Point_Inside.blend from the Blender folder.
2. Copy and paste the script from the evaluation folder (../point_inside_box.py).
3. Run it and evaluate the results.
   
```bash
$ /> cd Documents/GitHub/Collision_Detection/Blender/Collision_Detection
$ ../Collision_Detection/Blender> blender Point_Inside.blend
```

<p align="center">
<img src=https://github.com/rparak/Collision_Detection/blob/main/images/3D_Point_Tests.png width="800" height="350">
</p>

## 3D Line Intersections

A description of how to run a program to check if a line segment intersects with 3D primitive object.
1. Open Raycast.blend from the Blender folder.
2. Copy and paste the script from the evaluation folder (../raycast_box.py).
3. Run it and evaluate the results.
   
```bash
$ /> cd Documents/GitHub/Collision_Detection/Blender/Collision_Detection
$ ../Collision_Detection/Blender> blender Raycast.blend
```

<p align="center">
<img src=https://github.com/rparak/Collision_Detection/blob/main/images/3D_Line_Intersections.png width="800" height="350">
</p>

## Contact Info
Roman.Parak@outlook.com

## Citation (BibTex)
```bash
@misc{RomanParak_DataConverter,
  author = {Roman Parak},
  title = {An open-source collision detection library useful for robotics applications},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://https://github.com/rparak/Parametric_Curves}}
}
```

## License
[MIT](https://choosealicense.com/licenses/mit/)
