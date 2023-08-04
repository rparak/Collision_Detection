# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Dataclasses (Data Classes)
from dataclasses import dataclass, field
# Typing (Support for type hints)
import typing as tp
# Custom Library:
#   ../Lib/Transformation/Core
import Lib.Transformation.Core as Transformation 
#   ../Lib/Transformation/Utilities
import Lib.Transformation.Utilities.Mathematics as Mathematics

@dataclass
class Camera_Parameters_Str:
    """
    Description:
        The structure of the main parameters of the camera (sensor) object.

    Example:
        Initialization:
            Cls = Camera_Parameters_Str()
            Cls.T = ...
            ...
            Cls.Value = ..
    """

    # Homogeneous transformation matrix of the object.
    #   Unit [Matrix<float>]
    T: tp.List[tp.List[float]] = field(default_factory=list)
    #  The properties of the projection view.
    #   Unit [1: string, 2: float]
    #   1\ Projection of the camera's field of view: Perspective = ['PERSP'], Orthographic = ['ORTHO']
    Type: str = ''
    #   2\ Value is a PERSPECTIVE CAMERA LENS for perspective view, and ORTHOGRAPHIC CAMERA SCALE for ortographic view.
    Value: float = 0.0

"""
Camera view from the right.

    The properties of the projection view: Perspective, 50.0
"""
Right_View_Camera_Parameters_Str = Camera_Parameters_Str()
Right_View_Camera_Parameters_Str.T = Transformation.Homogeneous_Transformation_Matrix_Cls(None, np.float32).Rotation([Mathematics.Degree_To_Radian(70.0), 0.0, Mathematics.Degree_To_Radian(115.0)], 
                                                                                          'XYZ').Translation([3.25, 1.5, 1.45])
Right_View_Camera_Parameters_Str.Type  = 'PERSP'
Right_View_Camera_Parameters_Str.Value = 50.0

