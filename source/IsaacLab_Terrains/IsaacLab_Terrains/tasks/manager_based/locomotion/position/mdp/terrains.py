from isaaclab.utils import configclass
from isaaclab.terrains import SubTerrainBaseCfg
import trimesh
import numpy as np
from isaaclab.terrains.trimesh.mesh_terrains_cfg import MeshPlaneTerrainCfg
from isaaclab.terrains.trimesh.utils import make_box, make_plane




def custom_terrain(
    difficulty: float, cfg: MeshPlaneTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a flat terrain as a plane.

    .. image:: ../../_static/terrains/trimesh/flat_terrain.jpg
       :width: 45%
       :align: center

    Note:
        The :obj:`difficulty` parameter is ignored for this terrain.

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        A tuple containing the tri-mesh of the terrain and the origin of the terrain (in m).
    """
    # compute the position of the terrain
    origin = (cfg.size[0] / 2.0, cfg.size[1] / 2.0, 0.0)
    # compute the vertices of the terrain
    plane_mesh = make_plane(cfg.size, 0.0, center_zero=False)
    # return the tri-mesh and the position
    return [plane_mesh], np.array(origin)



@configclass
class CustomMeshCfg(SubTerrainBaseCfg):
    """Configuration for a custom mesh terrain."""

    function = custom_terrain
