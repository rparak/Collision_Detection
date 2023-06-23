# BPY (Blender as a python) [pip3 install bpy]
import bpy
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Typing (Support for type hints)
import typing as tp

def Deselect_All() -> None:
    """
    Description:
        Deselect all objects in the current scene.
    """
    
    for obj in bpy.context.selected_objects:
        bpy.data.objects[obj.name].select_set(False)

def Object_Exist(name: str) -> bool:
    """
    Description:
        Check if the object exists within the scene.
        
    Args:
        (1) name [string]: Object name.
        
    Returns:
        (1) parameter [bool]: 'True' if it exists, otherwise 'False'.
    """
    
    return True if bpy.context.scene.objects.get(name) else False

def Remove_Object(name: str) -> None:
    """
    Description:
        Remove the object (hierarchy) from the scene, if it exists. 

    Args:
        (1) name [string]: The name of the object.
    """

    # Find the object with the desired name in the scene.
    object_name = None
    for obj in bpy.data.objects:
        if name in obj.name and Object_Exist(obj.name) == True:
            object_name = obj.name
            break

    # If the object exists, remove it, as well as the other objects in the hierarchy.
    if object_name is not None:
        bpy.data.objects[object_name].select_set(True)
        for child in bpy.data.objects[object_name].children:
            child.select_set(True)
        bpy.ops.object.delete()
        bpy.context.view_layer.update()

def Set_Object_Material_Transparency(name: str, alpha: float) -> None:
    """
    Description:
        Set the transparency of the object material and/or the object hierarchy (if exists).
        
        Note: 
            alpha = 1.0: Render surface without transparency.
            
    Args:
        (1) name [string]: The name of the object.
        (2) alpha [float]: Transparency information.
                           (total transparency is 0.0 and total opacity is 1.0)
    """

    for obj in bpy.data.objects:
        if bpy.data.objects[name].parent == True:
            if obj.parent == bpy.data.objects[name]:
                for material in obj.material_slots:
                    if alpha == 1.0:
                        material.material.blend_method  = 'OPAQUE'
                    else:
                        material.material.blend_method  = 'BLEND'
                    
                    material.material.shadow_method = 'OPAQUE'
                    material.material.node_tree.nodes['Principled BSDF'].inputs['Alpha'].default_value = alpha
                
                # Recursive call.
                return Set_Object_Material_Transparency(obj.name, alpha)
        else:
            if obj == bpy.data.objects[name]:
                for material in obj.material_slots:
                    if alpha == 1.0:
                        material.material.blend_method  = 'OPAQUE'
                    else:
                        material.material.blend_method  = 'BLEND'
                    
                    material.material.shadow_method = 'OPAQUE'
                    material.material.node_tree.nodes['Principled BSDF'].inputs['Alpha'].default_value = alpha

def Set_Object_Material_Color(name: str, color: tp.List[float]):
    """
    Description:
        Set the material color of the individual object and/or the object hierarchy (if exists).
            
    Args:
        (1) name [string]: The name of the object.
        (2) color [Vector<float>]: RGBA color values: rgba(red, green, blue, alpha).
    """

    for obj in bpy.data.objects:
        if bpy.data.objects[name].parent == True:
            if obj.parent == bpy.data.objects[name]:
                for material in obj.material_slots:
                    material.material.node_tree.nodes['Principled BSDF'].inputs["Base Color"].default_value = color

                 # Recursive call.
                return Set_Object_Material_Color(obj.name, color)
        else:
            if obj == bpy.data.objects[name]:
                for material in obj.material_slots:
                    material.material.node_tree.nodes['Principled BSDF'].inputs["Base Color"].default_value = color 
                    
def __Add_Primitive(type: str, properties: tp.Tuple[float, tp.List[float], tp.List[float]]) -> bpy.ops.mesh:
    """
    Description:
        Add a primitive three-dimensional object.
        
    Args:
        (1) type [string]: Type of the object. 
                            Primitives: ['Plane', 'Cube', 'Sphere', 'Capsule']
        (2) properties [Dictionary {'Size/Radius': float, 'Scale/Size/None': Vector<float>, 
                                    'Location': Vector<float>]: Transformation properties of the created object. The structure depends 
                                                                on the specific object.
    
    Returns:
        (1) parameter [bpy.ops.mesh]: Individual three-dimensional object (primitive).
    """
        
    return {
        'Plane': lambda x: bpy.ops.mesh.primitive_plane_add(size=x['Size'], scale=x['Scale'], location=x['Location']),
        'Cube': lambda x: bpy.ops.mesh.primitive_cube_add(size=x['Size'], scale=x['Scale'], location=x['Location']),
        'Sphere': lambda x: bpy.ops.mesh.primitive_uv_sphere_add(radius=x['Radius'], location=x['Location']),
        'Capsule': lambda x: bpy.ops.mesh.primitive_round_cube_add(radius=x['Radius'], size=x['Size'], location=x['Location'], arc_div=10)
    }[type](properties)

def Create_Primitive(type: str, name: str, properties: tp.Tuple[tp.Tuple[float, tp.List[float]], tp.Tuple[float]]) -> None:
    """
    Description:
        Create a primitive three-dimensional object with additional properties.

    Args:
        (1) type [string]: Type of the object. 
                            Primitives: ['Plane', 'Cube', 'Sphere', 'Capsule']
        (2) name [string]: The name of the created object.
        (3) properties [{'transformation': {'Size/Radius': float, 'Scale/Size/None': Vector<float>, Location': Vector<float>}, 
                         'material': {'RGBA': Vector<float>, 'alpha': float}}]: Properties of the created object. The structure depends on 
                                                                                on the specific object.
    """

    # Create a new material and set the material color of the object.
    material = bpy.data.materials.new(f'{name}_mat')
    material.use_nodes = True
    material.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = properties['material']['RGBA']

    # Add a primitive three-dimensional object.
    __Add_Primitive(type, properties['transformation'])

    # Change the name and material of the object.
    bpy.context.active_object.name = name
    bpy.context.active_object.active_material = material

    # Set the transparency of the object material.
    if properties['material']['alpha'] < 1.0:
        Set_Object_Material_Transparency(name, properties['material']['alpha'])

    # Deselect all objects in the current scene.
    Deselect_All()

    # Update the scene.
    bpy.context.view_layer.update()

class Poly_3D(object):
    """
    Description:
        Visualization of a 3-D (dimensional) polyline.
        
    Initialization of the Class:
        Args:
            (1) name [string]: The name of the polyline (object).
            (2) curve_properties [Dictionary {'bevel_depth': float, 
                                              'color': Vector<float>}]: Properties of the curve (polyline).
                                                                        Note:
                                                                            bevel_depth: Radius of the bevel geometry.
                                                                            color: The color of the curve.
            (3) point_properties [Dictionary {'visibility': bool, 'radius': float, 'color': Vector<float>}]: Properties of curve points.
            
            Explanations:
                (2: bevel_depth)
                    Radius of the bevel geometry.
                (2: color)
                    The color of the curve.
                (3: visibility)
                    Visibility of points on the curve.
                (3: radius)
                    Radius of points.
                (3: color)
                    The color of the points.
    
        Example:
            Initialization:
                Cls = Poly_3D(name, {'bevel_depth': bevel_depth, 'color': color}, 
                              {'visibility': visibility, 'radius': radius, 'color': color})
            
            Features:
                # Remove the object (polyline) from the scene.
                Cls.Remove()
                
                # Initialize the size (length) of the polyline data set.
                Cls.Initialization(size_of_ds)
                
                # Add coordinates (x, y, z points) to the polyline.
                Cls.Add(index, coordinates)
                
                # Visualization of a 3-D (dimensional) polyline in the scene.
                Cls.Visualization()
                
                # Animation of a 3-D (dimensional) polyline in the scene.
                Cls.Animation(frame_start, frame_end)
    """

    def __init__(self, name: str, curve_properties: tp.Tuple[float, tp.List[float]], point_properties: tp.Tuple[tp.Union[None, bool], float, tp.List[float]]) -> None: 
        # << PRIVATE >> #
        self.__name = name
        self.__visibility_points = point_properties['visibility']
        # Remove the object (hierarchy) from the scene, if it exists
        Remove_Object(self.__name)
        # Create a polyline data-block.
        self.__data_block = bpy.data.curves.new(self.__name, type='CURVE')
        self.__data_block.dimensions = '3D'
        self.__data_block.bevel_depth = curve_properties['bevel_depth']
        # Radius of points.
        self.__radius_points = point_properties['radius']
        # The collection of splines in this polyline data-block.
        self.__polyline = self.__data_block.splines.new('POLY')
        # Creation of new material.
        #   Curve.
        self.__material_curve = bpy.data.materials.new(self.__data_block.name + 'mat')
        self.__material_curve.use_nodes = True
        self.__material_curve.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = curve_properties['color']
        #   Points (Sphere).
        if self.__visibility_points == True:
            self.__material_sphere = bpy.data.materials.new(self.__data_block.name + 'mat')
            self.__material_sphere.use_nodes = True
            self.__material_sphere.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = point_properties['color']
    
    def Initialization(self, size_of_ds: int) -> None:
        """
        Description:
            Initialize the size (length) of the polyline data set.
            
        Args:
            (1) size_of_ds [int]: The size (length) of the data set.
        """
        
        self.__polyline.points.add(size_of_ds - 1)
        
    def Add(self, index: int, coordinates: tp.List[float]) -> None:
        """
        Description:
            Add coordinates (x, y, z points) to the polyline.
            
        Args:
            (1) index [int]: Index of the current added point.
            (2) coordinates [Vector<float> 1x3]: Coordinates (x, y, z) of the polyline.
        """

        x, y, z = coordinates

        self.__polyline.points[index].co = (x, y, z, 1)

        if self.__visibility_points == True:
            self.__Add_Sphere(index, x, y, z)

    def __Add_Sphere(self, index: int, x: float, y: float, z: float) -> None:
        """
        Description:
            Add coordinates (x,y,z) of a spherical object (point).

        Args:
            (1) index [int]: Index of the current added point.
            (2 - 4) x, y, z [float]: Coordinates of the point (spherical object).
        """
        
        # Creation a spherical object with a defined position.
        Create_Primitive('Sphere', {'Radius': self.__radius_points, 'Location': (x, y, z)})
        bpy.ops.object.shade_smooth()
        # Change the name and material of the object.
        bpy.context.active_object.name = f'Point_{index}'
        bpy.context.active_object.active_material = self.__material_sphere
    
    def Visualization(self):
        """
        Description:
            Visualization of a 3-D (dimensional) polyline in the scene.
        """
        
        bpy.data.objects.new(self.__data_block.name, self.__data_block).data.materials.append(self.__material_curve)
        bpy.data.collections['Collection'].objects.link(bpy.data.objects.new(self.__data_block.name, self.__data_block))

        if self.__visibility_points == True:
            # Deselect all objects in the current scene.
            Deselect_All()
            # Selection of the main object (curve)
            bpy.ops.object.select_pattern(pattern=f'{self.__name}.*', extend=False)

            for i in range(len(self.__polyline.points)):
                # Connecting a child object (point_i .. n) to a parent object (curve).
                bpy.data.objects[f'Point_{i}'].parent = bpy.data.objects[f'{bpy.context.selected_objects[0].name}']

    def Animation(self, frame_start: float, frame_end: float) -> None:
        """
        Description:
            Animation of a 3-D (dimensional) polyline in the scene.
            
        Args:
            (1) frame_start, frame_end [float]: The frame (start, end) in which the keyframe is inserted.
        """
        
        # Specifies how the tart / end bevel factor maps to the spline.
        bpy.data.objects[self.__data_block.name].data.bevel_factor_mapping_start = 'SPLINE'
        bpy.data.objects[self.__data_block.name].data.bevel_factor_mapping_end   = 'SPLINE'
        
        # Factor that determines the location of the bevel:
        #   0.0 (0 %), 1.0 (100 %)
        bpy.data.objects[self.__data_block.name].data.bevel_factor_end = 0.0
        bpy.data.objects[self.__data_block.name].data.keyframe_insert(data_path='bevel_factor_end', frame=frame_start, index=-1)
        bpy.data.objects[self.__data_block.name].data.bevel_factor_end = 1.0
        bpy.data.objects[self.__data_block.name].data.keyframe_insert(data_path='bevel_factor_end', frame=frame_end, index=-1)