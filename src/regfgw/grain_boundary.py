import numpy as np
from dataclasses import dataclass
from pathlib import Path
from pymatgen.core import Structure, Lattice
from pymatgen.core.interface import GrainBoundary, GrainBoundaryGenerator, fix_pbc
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

@dataclass(frozen=True)
class GrainBoundaryParams:
    """Parameters defining a grain-boundary bicrystal"""
    rotation_axis: tuple[int, int, int] | tuple[int, int, int, int]
    gb_plane: tuple[int, int, int] | tuple[int, int, int, int]
    sigma: int | None = None
    rotation_angle: float | None = None
    normal: bool = True
    expand_times: int = 1
    rm_ratio: float = 0.3
    reference_species: tuple[str, ...] | None = None

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

@dataclass
class GrainSetup:
    """Parameters preparing grains required for bicrystal assembly"""
    top_grain: Structure
    bottom_grain: Structure
    initial_structure: Structure
    rotation_angle: float
    join_plane: tuple[int, int, int]
    oriented_unit_cell: Structure
    unit_ab_adjust: np.ndarray

class GrainBoundaryBuilder:
    """Build grain boundaries with explicit termination control."""
    def __init__(self, structure: Structure, params: GrainBoundaryParams):
        self.structure = structure
        self.params = params
        self.setup: GrainSetup | None = None

    def get_rotation_angle(self, generator: GrainBoundaryGenerator):
        """Get the grain-boundary rotation angle."""
        ratio = generator.get_ratio()
        rotation_angle = self.params.rotation_angle

        if rotation_angle is None:
            rotation_angles = generator.get_rotation_angle_from_sigma(
                sigma=self.params.sigma,
                r_axis=self.params.rotation_axis,
                lat_type=generator.lat_type,
                ratio=ratio,
            )
            if len(rotation_angles) == 0:
                raise ValueError(
                    f"No rotation angle found for "
                    f"sigma={self.params.sigma} and rotation_axis={self.params.rotation_axis}"
                )
            if len(rotation_angles) > 1:
                raise ValueError(
                    f"Multiple rotation angles found for "
                    f"sigma={self.params.sigma} and rotation_axis={self.params.rotation_axis}: "
                    f"{[float(angle) for angle in rotation_angles]}. Please specify an angle."
                )
            rotation_angle = rotation_angles[0]

        return float(rotation_angle)

    def extract_reference_structure(self):
        """Return the substructure containing only the reference species."""
        if self.params.reference_species is None:
            return self.structure.copy()

        if not self.params.reference_species:
            raise ValueError("reference_species cannot be empty.")

        available_species = {sp.symbol for sp in self.structure.types_of_species}
        missing_species = [symbol for symbol in self.params.reference_species if symbol not in available_species]

        if missing_species:
            raise ValueError(f"{missing_species} not present in the structure.")

        reference_structure = self.structure.copy()
        species_to_remove = [sp for sp in reference_structure.types_of_species if
                             sp.symbol not in self.params.reference_species]
        reference_structure.remove_species(species_to_remove)

        return reference_structure

    @staticmethod
    def get_crystal_transform(generator: GrainBoundaryGenerator):
        """Return the crystal transformation used by pymatgen."""
        trans_cry = np.eye(3)

        if generator.lat_type.lower() != "c":
            return trans_cry

        analyzer = SpacegroupAnalyzer(generator.initial_structure)
        conventional = analyzer.get_conventional_standard_structure()
        volume_ratio = generator.initial_structure.volume / conventional.volume

        if abs(volume_ratio - 0.5) < 1e-3:
            trans_cry = np.array(
                [
                    [0.5, 0.5, -0.5],
                    [-0.5, 0.5, 0.5],
                    [0.5, -0.5, 0.5],
                ]
            )
        elif abs(volume_ratio - 0.25) < 1e-3:
            trans_cry = np.array(
                [
                    [0.5, 0.5, 0.0],
                    [0.0, 0.5, 0.5],
                    [0.5, 0.0, 0.5],
                ]
            )

        return trans_cry

    def prepare(self, out_dir: str | Path | None = None):
        """Prepare oriented top and bottom grains for bicrystal assembly."""
        reference_structure = self.extract_reference_structure()
        generator = GrainBoundaryGenerator(reference_structure)
        rotation_angle = self.get_rotation_angle(generator)
        lat_type = generator.lat_type
        ratio = generator.get_ratio()
        trans_cry = self.get_crystal_transform(generator)
        top_transform, bottom_transform = generator.get_trans_mat(
            r_axis=self.params.rotation_axis,
            angle=rotation_angle,
            normal=self.params.normal,
            trans_cry=trans_cry,
            lat_type=lat_type,
            ratio=ratio,
            surface=self.params.gb_plane,
        )

        # Determine the joining plane following pymatgen.
        if lat_type != "c":
            if lat_type == "h":
                mu, mv = ratio
                trans_cry_join = np.array(
                    [
                        [1, 0, 0],
                        [-0.5, np.sqrt(3.0) / 2.0, 0],
                        [0, 0, np.sqrt(mu / mv)],
                    ]
                )
            elif lat_type == "r":
                mu, mv = ratio
                c2_a2_ratio = 3 / (2 - 6 * mv / mu)
                trans_cry_join = np.array(
                    [
                        [0.5, np.sqrt(3.0) / 6.0, np.sqrt(c2_a2_ratio) / 3.0],
                        [-0.5, np.sqrt(3.0) / 6.0, np.sqrt(c2_a2_ratio) / 3.0],
                        [0, -np.sqrt(3.0) / 3.0, np.sqrt(c2_a2_ratio) / 3.0],
                    ]
                )
            else:
                if lat_type == "t":
                    mu, mv = ratio
                    lam = mv
                elif lat_type == "o":
                    mu, lam, mv = (value for value in ratio)
                else:
                    raise RuntimeError(f"Unsupported lattice type: {lat_type}.")
                trans_cry_join = np.array(
                    [
                        [1, 0, 0],
                        [0, np.sqrt(lam / mv), 0],
                        [0, 0, np.sqrt(mu / mv)],
                    ]
                )
        else:
            trans_cry_join = trans_cry

        grain_matrix = bottom_transform @ trans_cry_join
        plane_init = np.cross(grain_matrix[0], grain_matrix[1])

        if lat_type != "c":
            plane_init = plane_init @ trans_cry_join.T

        plane_vector = (float(plane_init[0]), float(plane_init[1]), float(plane_init[2]))
        join_plane = generator.vec_to_surface(plane_vector)
        join_plane = (int(join_plane[0]), int(join_plane[1]), int(join_plane[2]))
        # Apply the transformations to the complete structure
        top_grain = fix_pbc(self.structure.copy() * top_transform)
        bottom_grain = fix_pbc(self.structure.copy() * bottom_transform, top_grain.lattice.matrix)

        # Obtain the smallest oriented unit cell and in-plane adjustment used by pymatgen
        if self.params.normal:
            temp_transform, _ = generator.get_trans_mat(
                r_axis=self.params.rotation_axis,
                angle=rotation_angle,
                normal=False,
                trans_cry=trans_cry,
                lat_type=lat_type,
                ratio=ratio,
                surface=self.params.gb_plane,
            )
            oriented_unit_cell = fix_pbc(self.structure.copy() * temp_transform)
            matrix = oriented_unit_cell.lattice.matrix
            normal_vector = np.cross(matrix[0], matrix[1])
            unit_normal = (normal_vector / np.linalg.norm(normal_vector))
            unit_ab_adjust = (matrix[2] - np.dot(unit_normal, matrix[2]) * unit_normal) / np.dot(unit_normal, matrix[2])
        else:
            oriented_unit_cell = top_grain.copy()
            unit_ab_adjust = np.zeros(3)

        self.setup = GrainSetup(
            top_grain=top_grain,
            bottom_grain=bottom_grain,
            initial_structure=self.structure,
            rotation_angle=rotation_angle,
            join_plane=join_plane,
            oriented_unit_cell=oriented_unit_cell,
            unit_ab_adjust=unit_ab_adjust,
        )

        if out_dir is not None:
            out_dir =Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            self.setup.top_grain.to(filename=out_dir / "top_grain.cif")
            self.setup.bottom_grain.to(filename=out_dir / "bottom_grain.cif")

        return self.setup

    @staticmethod
    def shift_cut_to_origin(structure: Structure, cut_c: float):
        """Move the selected fractional c termination to c=0."""
        if not 0.0 <= cut_c < 1.0:
            raise ValueError("cut_c must be in the interval [0, 1).")

        frac_coords = structure.frac_coords.copy()
        frac_coords[:, 2] = np.mod(frac_coords[:, 2] - cut_c, 1.0)
        frac_coords[np.isclose(frac_coords[:, 2], 1.0, atol=1e-10), 2] = 0.0
        frac_coords[np.isclose(frac_coords[:, 2], 0.0, atol=1e-10), 2] = 0.0

        return Structure(
            structure.lattice,
            structure.species,
            frac_coords,
            site_properties=structure.site_properties,
        )

    @staticmethod
    def remove_close_sites(structure: Structure, bond_length: float, rm_ratio: float):
        """Remove unrealistically close sites near the two grain boundaries."""
        lattice_matrix = structure.lattice.matrix
        normal_vector = np.cross(lattice_matrix[0], lattice_matrix[1])
        unit_normal = normal_vector / np.linalg.norm(normal_vector)
        cos_c_normal = np.dot(unit_normal, lattice_matrix[2]) / structure.lattice.c
        range_c = abs(bond_length) / cos_c_normal / structure.lattice.c
        near_sites = []
        far_sites = []

        for site in structure:
            c = site.frac_coords[2]
            if c < range_c or c > 1.0 - range_c or abs(c - 0.5) < range_c:
                near_sites.append(site)
            else:
                far_sites.append(site)

        if not near_sites:
            return structure

        near_structure = Structure.from_sites(near_sites)
        near_structure.merge_sites(tol=bond_length * rm_ratio, mode="delete")
        grain_boundary = Structure.from_sites(far_sites + near_structure.sites)

        return fix_pbc(grain_boundary, structure.lattice.matrix)

    def build(
            self,
            top_cut_c: float,
            bottom_cut_c: float,
            ab_shift: tuple[float, float] = (0.0, 0.0),
            out_dir: str | Path | None = None,
    ):
        """Assemble a periodic bicrystal from prepared grains."""
        if self.setup is None:
            raise RuntimeError("prepare() must be called before build().")

        setup = self.setup
        top_grain = self.shift_cut_to_origin(setup.top_grain, top_cut_c)
        bottom_grain = self.shift_cut_to_origin(setup.bottom_grain, bottom_cut_c)
        # Determine the relative c adjustment for bicrystal assembly.
        edge_top = 1.0 - np.max(top_grain.frac_coords[:, 2])
        edge_bottom = 1.0 - np.max(bottom_grain.frac_coords[:, 2])
        c_adjust = (edge_top - edge_bottom) / 2.0
        half_lattice = top_grain.lattice
        # Construct the periodic bicrystal lattice
        whole_matrix = half_lattice.matrix.copy()
        whole_matrix[2] = 2.0 * half_lattice.matrix[2]
        whole_lattice = Lattice(whole_matrix)
        # Place the top grain above the bottom grain
        c_translation = half_lattice.matrix[2] * (1 + c_adjust)
        top_translation = c_translation + setup.unit_ab_adjust * np.linalg.norm(c_translation)
        top_translation += (ab_shift[0] * whole_matrix[0] + ab_shift[1] * whole_matrix[1])
        species = bottom_grain.species + top_grain.species
        coords = [site.coords for site in bottom_grain] + [site.coords + top_translation for site in top_grain]
        grain_labels = ["bottom"] * len(bottom_grain) + ["top"] * len(top_grain)
        gb = Structure(
            whole_lattice,
            species,
            coords,
            coords_are_cartesian=True,
            site_properties={"grain_label": grain_labels},
        )
        # Use the prepared grain to define the bulk nearest-neighbour scale.
        distance_matrix = top_grain.distance_matrix
        distances = distance_matrix[distance_matrix > 1e-8]
        bond_length = float(np.min(distances))
        gb = self.remove_close_sites(gb, bond_length, self.params.rm_ratio)
        gb = GrainBoundary(
            lattice=gb.lattice,
            species=gb.species,
            coords=gb.cart_coords,
            rotation_axis=self.params.rotation_axis,
            rotation_angle=setup.rotation_angle,
            gb_plane=self.params.gb_plane,
            join_plane=setup.join_plane,
            init_cell=setup.initial_structure,
            vacuum_thickness=0.0,
            ab_shift=ab_shift,
            site_properties=gb.site_properties,
            oriented_unit_cell=setup.oriented_unit_cell,
            coords_are_cartesian=True,
        )

        if out_dir is not None:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            gb.to(filename=out_dir / "grain_boudary.cif")

        return gb
