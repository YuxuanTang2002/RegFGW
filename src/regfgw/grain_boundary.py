from pathlib import Path
from dataclasses import dataclass
from pymatgen.core import Structure
from pymatgen.core.interface import GrainBoundaryGenerator

@dataclass
class GrainBoundaryParams:
    """Parameters defining a grain-boundary bicrystal."""
    rotation_axis: tuple[int, int, int]
    gb_plane: tuple[int, int, int]
    sigma: int | None = None
    rotation_angle: float | None = None
    expand_times: int = 1
    ab_shift: tuple[float, float] = (0.25191063, 0.0)
    rm_ratio: float = 0.4

    def __post_init__(self):
        if all(index == 0 for index in self.rotation_axis):
            raise ValueError("rotation_axis cannot be (0, 0, 0).")

        if all(index == 0 for index in self.gb_plane):
            raise ValueError("gb_plane cannot be (0, 0, 0).")

        if (self.sigma is None) and (self.rotation_angle is None):
            raise ValueError("Either sigma or rotation_angle must be provided.")

        if (self.sigma is not None) and (self.sigma < 1):
            raise ValueError("sigma must be at least 1.")

        if self.expand_times < 1:
            raise ValueError("expand_times must be at least 1.")

def build_grain_boundary(structure: Structure, params: GrainBoundaryParams):
    """Construct a periodic grain-boundary bicrystal from a bulk structure."""
    generator = GrainBoundaryGenerator(structure)
    ratio = generator.get_ratio()
    rotation_angle = params.rotation_angle

    if rotation_angle is None:
        rotation_angles = generator.get_rotation_angle_from_sigma(
            sigma=params.sigma,
            r_axis=params.rotation_axis,
            ratio=ratio,
        )
        if len(rotation_angles) == 0:
            raise ValueError(
                f"No rotation angle found for "
                f"sigma={params.sigma} and rotation_axis={params.rotation_axis}"
            )
        if len(rotation_angles) > 1:

            raise ValueError(
                f"Multiple rotation angles found for "
                f"sigma={params.sigma} and rotation_axis={params.rotation_axis}: "
                f"{[float(angle) for angle in rotation_angles]}. Please specify an angle."
            )
        rotation_angle = rotation_angles[0]

    return generator.gb_from_parameters(
        rotation_axis=params.rotation_axis,
        rotation_angle=rotation_angle,
        plane=params.gb_plane,
        expand_times=params.expand_times,
        ab_shift=params.ab_shift,
        rm_ratio=params.rm_ratio,
        ratio=ratio,
        normal=True,
    )

def main():
    structure = Structure.from_file("data/cells/Li7La3Zr2O12.cif")
    zr_structure = structure.copy()
    params = GrainBoundaryParams(
        rotation_axis=(1, -1, 0),
        rotation_angle=109.47122063,
        # sigma=3,
        gb_plane=(1, 1, 2),
    )
    grain_boundary = build_grain_boundary(
        structure=zr_structure,
        params=params,
    )
    Path("results").mkdir(parents=True, exist_ok=True)
    grain_boundary.to("results/llzo_sigma3_112.cif")

if __name__ == "__main__":
    main()