from pathlib import Path
from dataclasses import dataclass
from functools import reduce
import math

import numpy as np

from pymatgen.core import Lattice, Structure
from pymatgen.core.interface import GrainBoundary, GrainBoundaryGenerator, fix_pbc
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


@dataclass
class GrainBoundaryParams:
    """Parameters defining a grain-boundary bicrystal."""

    rotation_axis: tuple[int, int, int]
    gb_plane: tuple[int, int, int]
    sigma: int | None = None
    rotation_angle: float | None = None
    expand_times: int = 1
    rm_ratio: float = 0.3
    reference_species: str | tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if all(index == 0 for index in self.rotation_axis):
            raise ValueError("rotation_axis cannot be (0, 0, 0).")

        if all(index == 0 for index in self.gb_plane):
            raise ValueError("gb_plane cannot be (0, 0, 0).")

        if self.sigma is None and self.rotation_angle is None:
            raise ValueError("Either sigma or rotation_angle must be provided.")

        if self.sigma is not None and self.sigma < 1:
            raise ValueError("sigma must be at least 1.")

        if self.expand_times < 1:
            raise ValueError("expand_times must be at least 1.")


@dataclass
class GrainSetup:
    """Oriented grains and metadata required for later bicrystal assembly."""

    top_grain: Structure
    bottom_grain: Structure
    oriented_unit_cell: Structure
    rotation_axis: tuple[int, int, int]
    rotation_angle: float
    gb_plane: tuple[int, int, int]
    join_plane: tuple[int, int, int]
    unit_ab_adjust: np.ndarray


def _resolve_rotation_angle(
    generator: GrainBoundaryGenerator,
    params: GrainBoundaryParams,
) -> float:
    if params.rotation_angle is not None:
        return float(params.rotation_angle)

    rotation_angles = generator.get_rotation_angle_from_sigma(
        sigma=params.sigma,
        r_axis=params.rotation_axis,
        ratio=generator.get_ratio(),
    )

    if len(rotation_angles) == 0:
        raise ValueError(
            f"No rotation angle found for sigma={params.sigma} and "
            f"rotation_axis={params.rotation_axis}."
        )

    if len(rotation_angles) > 1:
        raise ValueError(
            f"Multiple rotation angles found for sigma={params.sigma} and "
            f"rotation_axis={params.rotation_axis}: "
            f"{[round(float(angle), 6) for angle in rotation_angles]}. "
            "Please specify rotation_angle explicitly."
        )

    return float(rotation_angles[0])


def _reference_species_tuple(
    reference_species: str | tuple[str, ...],
) -> tuple[str, ...]:
    """Normalize one or more reference species to a non-empty tuple."""
    if isinstance(reference_species, str):
        species = (reference_species,)
    else:
        species = tuple(reference_species)

    if not species:
        raise ValueError("reference_species cannot be empty.")

    return species


def _reference_structure(
    structure: Structure,
    reference_species: str | tuple[str, ...],
) -> Structure:
    """Return the substructure containing only the selected reference species."""
    species = _reference_species_tuple(reference_species)
    available = {sp.symbol for sp in structure.types_of_species}
    missing = [symbol for symbol in species if symbol not in available]
    if missing:
        raise ValueError(
            f"Reference species not present in structure: {missing}."
        )

    reference = structure.copy()
    to_remove = [
        sp for sp in reference.types_of_species if sp.symbol not in species
    ]
    reference.remove_species(to_remove)

    if len(reference) == 0:
        raise ValueError("The selected reference substructure is empty.")

    return reference


def _cubic_trans_cry(generator: GrainBoundaryGenerator) -> np.ndarray:
    """Mirror pymatgen's cubic primitive-cell handling."""
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


def _shift_cut_c_to_origin(structure: Structure, cut_c: float) -> Structure:
    """Move a user-selected fractional-c plane to c = 0 for joining.

    With ``normal=True``, constant fractional-c surfaces are parallel to the
    grain-boundary plane. The selected coordinate may pass through an atomic
    layer or lie between layers.
    """
    frac = np.array(structure.frac_coords, dtype=float)
    frac[:, 2] = np.mod(frac[:, 2] - cut_c, 1.0)
    frac[np.isclose(frac, 1.0, atol=1e-10)] = 0.0
    frac[np.isclose(frac, 0.0, atol=1e-10)] = 0.0

    return Structure(
        structure.lattice,
        structure.species,
        frac,
        site_properties=structure.site_properties,
    )


def build_oriented_grains(
    structure: Structure,
    params: GrainBoundaryParams,
) -> GrainSetup:
    """Build the default oriented top and bottom grains.

    The selected reference substructure (one or more species) defines the CSL
    transformations. The complete structure is transformed with the same t1/t2
    matrices. No cut-plane choice is made here; that is deliberately deferred to
    the bicrystal-assembly stage.
    """
    if params.reference_species is None:
        raise ValueError(
            "reference_species must specify one or more species for the reference structure."
        )

    reference_species = _reference_species_tuple(params.reference_species)
    reference = _reference_structure(structure, reference_species)
    reference_generator = GrainBoundaryGenerator(reference)
    rotation_angle = _resolve_rotation_angle(reference_generator, params)
    ratio = reference_generator.get_ratio()
    lat_type = reference_generator.lat_type.lower()

    if lat_type != "c":
        raise NotImplementedError(
            "reference_species construction is currently implemented for cubic systems."
        )

    rotation_axis = tuple(params.rotation_axis)
    axis_gcd = reduce(math.gcd, rotation_axis)
    if axis_gcd != 1:
        rotation_axis = tuple(round(x / axis_gcd) for x in rotation_axis)

    gb_plane = tuple(params.gb_plane)
    plane_gcd = reduce(math.gcd, gb_plane)
    if plane_gcd != 1:
        gb_plane = tuple(round(x / plane_gcd) for x in gb_plane)

    trans_cry = _cubic_trans_cry(reference_generator)

    # The selected reference substructure alone defines the CSL transformations.
    t1, t2 = reference_generator.get_trans_mat(
        r_axis=rotation_axis,
        angle=rotation_angle,
        normal=True,
        trans_cry=trans_cry,
        lat_type=lat_type,
        ratio=ratio,
        surface=gb_plane,
    )

    grain_matrix = np.dot(t2, trans_cry)
    plane_init = np.cross(grain_matrix[0], grain_matrix[1])
    join_plane = reference_generator.vec_to_surface(plane_init)

    ref_parent = reference_generator.initial_structure.copy()
    full_parent = structure.copy()

    ref_top = fix_pbc(ref_parent * t1)
    ref_bottom = fix_pbc(ref_parent * t2, ref_top.lattice.matrix)
    full_top = fix_pbc(full_parent * t1)
    full_bottom = fix_pbc(full_parent * t2, full_top.lattice.matrix)

    # Same oriented-unit-cell construction used by pymatgen for normal=True.
    t_temp = reference_generator.get_trans_mat(
        r_axis=rotation_axis,
        angle=rotation_angle,
        normal=False,
        trans_cry=trans_cry,
        lat_type=lat_type,
        ratio=ratio,
        surface=gb_plane,
    )
    oriented_unit_cell = fix_pbc(full_parent * t_temp[0])
    t_matrix = oriented_unit_cell.lattice.matrix
    normal_v_plane = np.cross(t_matrix[0], t_matrix[1])
    unit_normal_v = normal_v_plane / np.linalg.norm(normal_v_plane)
    unit_ab_adjust = (
        t_matrix[2] - np.dot(unit_normal_v, t_matrix[2]) * unit_normal_v
    ) / np.dot(unit_normal_v, t_matrix[2])

    # Determine pymatgen-style coincidence labels from the selected reference grains.
    n_ref = len(ref_top)
    ref_pair = Structure(
        ref_top.lattice,
        ref_top.species + ref_bottom.species,
        list(ref_top.frac_coords) + list(ref_bottom.frac_coords),
    )
    ref_distances = ref_pair.lattice.get_all_distances(
        ref_pair.frac_coords[:n_ref],
        ref_pair.frac_coords[n_ref : 2 * n_ref],
    )
    incident = np.nonzero(ref_distances < np.min(ref_distances) + 1e-8)

    ref_top_incident = {
        (
            ref_top[i].specie.symbol,
            tuple(np.round(ref_top.frac_coords[i] % 1.0, 8)),
        )
        for i in incident[0]
    }
    ref_bottom_incident = {
        (
            ref_bottom[i].specie.symbol,
            tuple(np.round(ref_bottom.frac_coords[i] % 1.0, 8)),
        )
        for i in incident[1]
    }

    def labels_for(
        grain: Structure,
        side: str,
        incident_sites: set[tuple[str, tuple[float, ...]]],
    ) -> list[str]:
        labels = []
        for site in grain:
            key = (
                site.specie.symbol,
                tuple(np.round(site.frac_coords % 1.0, 8)),
            )
            if site.specie.symbol in reference_species and key in incident_sites:
                labels.append(f"{side}_incident")
            else:
                labels.append(side)
        return labels

    full_top = Structure(
        full_top.lattice,
        full_top.species,
        full_top.frac_coords,
        site_properties={
            "grain_label": labels_for(full_top, "top", ref_top_incident)
        },
    )
    full_bottom = Structure(
        full_bottom.lattice,
        full_bottom.species,
        full_bottom.frac_coords,
        site_properties={
            "grain_label": labels_for(full_bottom, "bottom", ref_bottom_incident)
        },
    )

    # Expand reference and complete grains identically.
    ref_top.make_supercell([1, 1, params.expand_times])
    ref_bottom.make_supercell([1, 1, params.expand_times])
    full_top.make_supercell([1, 1, params.expand_times])
    full_bottom.make_supercell([1, 1, params.expand_times])

    ref_top = fix_pbc(ref_top)
    ref_bottom = fix_pbc(ref_bottom)
    full_top = fix_pbc(full_top)
    full_bottom = fix_pbc(full_bottom)

    return GrainSetup(
        top_grain=full_top,
        bottom_grain=full_bottom,
        oriented_unit_cell=oriented_unit_cell,
        rotation_axis=rotation_axis,
        rotation_angle=rotation_angle,
        gb_plane=gb_plane,
        join_plane=join_plane,
        unit_ab_adjust=unit_ab_adjust,
    )


def build_bicrystal(
    structure: Structure,
    params: GrainBoundaryParams,
    setup: GrainSetup,
    top_cut_c: float,
    bottom_cut_c: float,
    ab_shift: tuple[float, float],
) -> GrainBoundary:
    """Assemble one bicrystal from user-selected fractional-c cut positions.

    ``normal=True`` is fixed in the CSL construction, so constant fractional-c
    surfaces are parallel to the grain-boundary plane. The selected coordinates
    are moved to c = 0 before the two complete grains are joined.
    """
    if not 0.0 <= top_cut_c < 1.0:
        raise ValueError("top_cut_c must be in the interval [0, 1).")
    if not 0.0 <= bottom_cut_c < 1.0:
        raise ValueError("bottom_cut_c must be in the interval [0, 1).")

    top = _shift_cut_c_to_origin(setup.top_grain, top_cut_c)
    bottom = _shift_cut_c_to_origin(setup.bottom_grain, bottom_cut_c)

    # The complete chemical structures determine the normal placement.
    edge_b = 1.0 - max(bottom.frac_coords[:, 2])
    edge_t = 1.0 - max(top.frac_coords[:, 2])
    c_adjust = (edge_t - edge_b) / 2.0

    half_lattice = top.lattice
    normal_v_plane = np.cross(half_lattice.matrix[0], half_lattice.matrix[1])
    unit_normal_v = normal_v_plane / np.linalg.norm(normal_v_plane)

    whole_matrix = np.array(half_lattice.matrix)
    whole_matrix[2] = half_lattice.matrix[2] * 2
    whole_lattice = Lattice(whole_matrix)

    all_species = [site.specie for site in bottom]
    all_species.extend(site.specie for site in top)

    grain_labels = (
        bottom.site_properties["grain_label"]
        + top.site_properties["grain_label"]
    )

    all_coords = [site.coords for site in bottom]
    all_coords.extend(
        site.coords
        + half_lattice.matrix[2] * (1.0 + c_adjust)
        + setup.unit_ab_adjust
        * np.linalg.norm(half_lattice.matrix[2] * (1.0 + c_adjust))
        + ab_shift[0] * whole_matrix[0]
        + ab_shift[1] * whole_matrix[1]
        for site in top
    )

    gb = Structure(
        whole_lattice,
        all_species,
        all_coords,
        coords_are_cartesian=True,
        site_properties={"grain_label": grain_labels},
    )

    # Same close-site deletion strategy as pymatgen. The distance scale comes
    # from the complete bulk structure because the final GB contains all species.
    distance = structure.distance_matrix
    bond_length = float(np.min(distance[np.nonzero(distance)]))

    cos_c_norm_plane = np.dot(unit_normal_v, whole_matrix[2]) / whole_lattice.c
    range_c_len = abs(bond_length / cos_c_norm_plane / whole_lattice.c)

    sites_near_gb = []
    sites_away_gb = []
    for site in gb:
        c = site.frac_coords[2]
        if (
            c < range_c_len
            or c > 1 - range_c_len
            or 0.5 - range_c_len < c < 0.5 + range_c_len
        ):
            sites_near_gb.append(site)
        else:
            sites_away_gb.append(site)

    if sites_near_gb:
        near_gb = Structure.from_sites(sites_near_gb)
        near_gb.merge_sites(
            tol=bond_length * params.rm_ratio,
            mode="delete",
        )
        gb = Structure.from_sites(sites_away_gb + near_gb.sites)

    gb = fix_pbc(gb, whole_lattice.matrix)

    return GrainBoundary(
        whole_lattice,
        gb.species,
        gb.cart_coords,
        setup.rotation_axis,
        setup.rotation_angle,
        setup.gb_plane,
        setup.join_plane,
        structure,
        0.0,
        ab_shift,
        site_properties=gb.site_properties,
        oriented_unit_cell=setup.oriented_unit_cell,
        coords_are_cartesian=True,
    )


def _prompt_cut_c(label: str) -> float:
    while True:
        value = input(
            f"Enter the {label} grain cut position as fractional c in [0, 1): "
        ).strip()
        try:
            cut_c = float(value)
        except ValueError:
            print("Please enter a valid floating-point value.")
            continue

        if 0.0 <= cut_c < 1.0:
            return cut_c

        print("The fractional c coordinate must be in the interval [0, 1).")


def _prompt_ab_shift() -> tuple[float, float]:
    while True:
        value = input("Enter ab shift as two fractional values (e.g. 0.0 0.0): ").strip()
        parts = value.replace(",", " ").split()
        if len(parts) != 2:
            print("Please enter exactly two numbers.")
            continue

        try:
            shift_a, shift_b = (float(parts[0]), float(parts[1]))
        except ValueError:
            print("Please enter valid floating-point values.")
            continue

        return shift_a, shift_b


def main() -> None:
    structure = Structure.from_file("data/cells/Li7La3Zr2O12.cif")

    params = GrainBoundaryParams(
        rotation_axis=(1, -1, 0),
        rotation_angle=109.47122063,
        gb_plane=(1, 1, 2),
        reference_species=("Zr",),  # e.g. ("Zr", "La") for a multi-species reference
    )

    # Stage 1: construct and export the default oriented complete grains.
    setup = build_oriented_grains(
        structure=structure,
        params=params,
    )

    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    top_path = output_dir / "llzo_sigma3_112_top_grain.cif"
    bottom_path = output_dir / "llzo_sigma3_112_bottom_grain.cif"
    setup.top_grain.to(top_path)
    setup.bottom_grain.to(bottom_path)

    print("\nOriented grains written for visual inspection:")
    print(f"  Top:    {top_path}")
    print(f"  Bottom: {bottom_path}")

    print(
        "\nInspect the two grain CIF files in VESTA/OVITO. "
        "Choose the fractional-c positions at which the top and bottom grains "
        "should be joined. The cut positions may pass through an atomic plane "
        "or lie between planes."
    )

    # Stage 2: user chooses arbitrary c-origin/cut phases and the in-plane registry.
    top_cut_c = _prompt_cut_c("top")
    bottom_cut_c = _prompt_cut_c("bottom")
    ab_shift = _prompt_ab_shift()

    grain_boundary = build_bicrystal(
        structure=structure,
        params=params,
        setup=setup,
        top_cut_c=top_cut_c,
        bottom_cut_c=bottom_cut_c,
        ab_shift=ab_shift,
    )

    gb_path = output_dir / "llzo_sigma3_112.cif"
    grain_boundary.to(gb_path)

    print("\nSelected bicrystal parameters:")
    print(f"  reference_species = {params.reference_species}")
    print(f"  top_cut_c          = {top_cut_c:.8f}")
    print(f"  bottom_cut_c       = {bottom_cut_c:.8f}")
    print(f"  ab_shift           = ({ab_shift[0]:.8f}, {ab_shift[1]:.8f})")
    print(f"  output             = {gb_path}")


if __name__ == "__main__":
    main()
