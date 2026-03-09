import os
import json
import numpy as np
from monty.json import MontyEncoder
from numpy.linalg import LinAlgError
from dataclasses import dataclass
from pymatgen.core import Structure
from pymatgen.core import Lattice
from pymatgen.core.surface import get_symmetrically_distinct_miller_indices
from pymatgen.core.interface import Interface
from pymatgen.analysis.interfaces.zsl import ZSLGenerator, vec_area
from pymatgen.analysis.interfaces.coherent_interfaces import CoherentInterfaceBuilder
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from .structure_to_graph import GraphEncoder

# -----------------------------------------------------------------------------
# Parameter containers
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ZSLParams:
    """
    Parameters control Zur-McGill lattice(ZSL) matching for coherent interfaces.

    Attributes
    ----------
    max_area: Maximum allowed in-plane coincidence supercell area(Å^2)
    max_area_ratio_tol: Relative tolerance for film/substrate in-plane area ratio
    max_length_tol: Relative tolerance for matching in-plane lattice vector lengths
    max_angle_tol: Absolute tolerance for matching in-plane lattice vector angles
    """
    max_area: float = 150.0
    max_area_ratio_tol: float = 0.06
    max_length_tol: float = 0.03
    max_angle_tol: float = 0.02

@dataclass(frozen=True)
class InterfaceParams:
    """
    Geometric parameters for building coherent interfaces

    Attributes
    ----------
    film_layers, substrate_layers: Slab thickness controls. One layer corresponds to the minimum stacking repeat period.
    vacuum: Vacuum thickness above film slab(Å)
    """
    film_layers: int = 3
    substrate_layers: int = 3
    gap: float = 5.0
    vacuum: float = 20.0

# -----------------------------------------------------------------------------
# Main builder
# -----------------------------------------------------------------------------

class InterfaceBuilder:
    """
    Coherent interface builder based on pymatgen ZSL + CoherentInterfaceBuilder

    Workflow
    --------
    1) Enumerate symmetrically distinct Miller indices for substrate and film.
    2) Build a CoherentInterfaceBuilder(CIB) for each Miller index pair.
    3) Loop over terminations and build interface candidates.
    4) (Optional) Build orientation-consistent bulk references(film/substrate slabs) for bulk-based descriptors.
    """
    def __init__(
            self,
            substrate: Structure, film: Structure,
            max_miller_idx=1,
            zsl_params=ZSLParams(),
            interface_params=InterfaceParams(),
    ):
        """
        Parameters
        ----------
        substrate, film: Bulk unit cells(pymatgen Structure) forming the two sides of the interface
        max_miller_idx: Maximum Miller index for enumerating distinct facets
        zsl_params: ZSL lattice matching tolerances
        interface_params: Slab thickness, gap and vacuum used for interface construction
        """
        self.substrate = self.standarize_struct(substrate)
        self.film = self.standarize_struct(film)
        self.max_miller_idx = max_miller_idx
        self.zsl_params = zsl_params
        self.interface_params = interface_params

        # Enumerate symmetrically distinct Miller indices to reduce redundant facets.
        self.s_indices = get_symmetrically_distinct_miller_indices(self.substrate, max_index=self.max_miller_idx)
        self.f_indices = get_symmetrically_distinct_miller_indices(self.film, max_index=self.max_miller_idx)

        # ZSL generator defines admissible coincidence lattices.
        self.zsl = ZSLGenerator(
            max_area=self.zsl_params.max_area,
            max_area_ratio_tol=self.zsl_params.max_area_ratio_tol,
            max_length_tol=self.zsl_params.max_length_tol,
            max_angle_tol=self.zsl_params.max_angle_tol,
        )

    # -------------------------------------------------------------------------
    # Geometry utilities
    # -------------------------------------------------------------------------

    @staticmethod
    def standarize_struct(struct: Structure):
        try:
            sga = SpacegroupAnalyzer(struct, symprec=1e-2, angle_tolerance=5.0)
            std = sga.get_conventional_standard_structure()
            if std is None:
                print("[WARN] spglib standarization returned None. Proceeding with original structure.")
                return struct
            return std
        except Exception as e:
            print(f"[WARN] spalib standarization raised {type(e).__name__}: {e}. Proceeding with original structure.")
            return struct

    @staticmethod
    def interface_area(interface: Structure):
        """
        Compute the ab in-plane area of an interface structure.

        Notes
        -----
        Pymatgen slab convention: a, b are in-plane, c is surface normal(z)
        """
        a_vec, b_vec = interface.lattice.matrix[0], interface.lattice.matrix[1]
        area = float(vec_area(a_vec, b_vec))
        return area

    @staticmethod
    def extract_thickness(structure: Structure):
        """Estimate slab thickness by z span of cartesian coordinates."""
        z = np.asarray(structure.cart_coords, dtype=float)[:, 2]
        return float(z.max() - z.min())

    def recenter_slab(self, structure: Structure):
        """Recenter a slab structure along the z direction by trimming vacuum and redefining the lattice c vector."""
        s = structure.copy()
        cart = np.asarray(s.cart_coords, dtype=float)
        z = cart[:, 2]
        z_min, z_max = float(z.min()), float(z.max())
        thickness = z_max - z_min

        if thickness <= 1e-6:
            raise ValueError("Structure thickness is too small.")

        padding = float(self.interface_params.vacuum / 2.0)
        cart[:, 2] += (-z_min + padding)
        a, b = s.lattice.matrix[0], s.lattice.matrix[1]
        c = np.array([0.0, 0.0, thickness + 2.0 * padding], dtype=float)
        lattice = Lattice(np.vstack([a, b, c]))

        return Structure(
            lattice=lattice,
            species=s.species,
            coords=cart,
            coords_are_cartesian=True
        )

    # -------------------------------------------------------------------------
    # CIB construction
    # -------------------------------------------------------------------------

    def build_cib(self, substrate_miller, film_miller):
        """
        Build a cib for a given Miller index pair.

        Returns
        -------
        None if pymatgen raises ValueError (e.g., invalid slab construction,
        no valid matching, or geometry constraints violated).
        """
        try:
            cib = CoherentInterfaceBuilder(
                substrate_structure=self.substrate, film_structure=self.film,
                substrate_miller=substrate_miller, film_miller=film_miller,
                zslgen=self.zsl,
            )
            return cib
        except ValueError:
            return None

    def get_interfaces(
            self,
            cib: CoherentInterfaceBuilder,
            term,
            film_layers: int | None = None,
            substrate_layers: int | None = None,
    ):
        """
        Generate coherent interface candidates for a given termination pair.

        Parameters
        ----------
        cib: CoherentInterfaceBuilder associated with a fixed (substrate_miller, film_miller)
        term: Termination pair from 'cib.terminations' (film_term, substrate_term)
        film_layers, substrate_layers: int or None. If None, use defaults from InterfaceParams.

        Returns
        -------
        list[Structure]: All candidates returned by 'cib.get_interface()' for specific termination
        """
        if film_layers is None:
            film_layers = self.interface_params.film_layers

        if substrate_layers is None:
            substrate_layers = self.interface_params.substrate_layers

        gap = self.interface_params.gap

        try:
            interfaces = list(cib.get_interfaces(
                term,
                gap=gap,
                vacuum_over_film=self.interface_params.vacuum,
                substrate_thickness=substrate_layers,
                film_thickness=film_layers,
            ))
        except LinAlgError as e:
            print(
                "[WARN] Skipping interface candidates due to LinAlgError (likely singular supercell transform). "
                f"{type(e).__name__}: {e}"
            )
            return []

        return interfaces

    def collect_candidates(
            self,
            cib: CoherentInterfaceBuilder,
            term,
            film_layers: int | None = None,
            substrate_layers: int | None = None,
    ):
        """
        Collect all interface candidates and their in-plane areas.

        Returns
        -------
        list[tuple(Structure, float)]: Each item is (interface, area)
        """
        candidates = []

        for itf in self.get_interfaces(cib, term, film_layers=film_layers, substrate_layers=substrate_layers):
            area = self.interface_area(itf)
            candidates.append((itf, float(area)))

        return candidates

    # -------------------------------------------------------------------------
    # Flip enumeration
    # -------------------------------------------------------------------------

    @staticmethod
    def neg_miller(m):
        return tuple(int(-i) for i in m)

    @staticmethod
    def mirror_along_normal(f, indices: list[int]):
        """Return new frac_coords with selected indices mirrored along c about the mid-plane."""
        fc = f[indices, 2]
        fc_mid = 0.5 * (float(fc.min()) + float(fc.max()))
        f[indices, 2] = 2.0 * fc_mid - fc
        return f

    def flip_interface(self, itf: Interface, flip_sub=False, flip_film=False):
        """Rebuild a new Interface object after flipping substrate or/and film."""
        f = np.array(itf.frac_coords, dtype=float, copy=True)

        if flip_sub:
            f = self.mirror_along_normal(f, itf.substrate_indices)

        if flip_film:
            f =  self.mirror_along_normal(f, itf.film_indices)

        f[:, :2] = f[:, :2] % 1.0

        return Interface(
            lattice=itf.lattice,
            species=itf.species,
            coords=f,
            coords_are_cartesian=False,
            site_properties=itf.site_properties,
            gap=itf.gap,
            vacuum_over_film=itf.vacuum_over_film,
        )

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def get_interface_records(
            self, substrate_miller, film_miller, term,
            build_bulk_refs=True, structure_check=False,
            cib: CoherentInterfaceBuilder | None = None,
    ):
        """
        Build full candidate records for a given (substrate_miller, film_miller, termination).
        Enumerate flips tp avoid missing termination combinations.

        Returns
        -------
        list[dict] or None
        * substrate_miller
        * film_miller
        * termination
        * cand_id: 0, ..., N-1
        * interface: Structure
        * area: Å^2
        * substrate_bulk(optional): Structure
        * film_bulk(Optional): Structure
        * sub_period_layers / film_period_layers: int: the number of z-coplanar atomic layers
        comprising one minimal stacking repeat period along the surface normal
        """
        if cib is None:
            cib = self.build_cib(substrate_miller=substrate_miller, film_miller=film_miller)

        if cib is None:
            return None

        itfs = self.collect_candidates(cib, term)

        if not itfs:
            return None

        records = []
        # Enumerate flip variants.
        flip_specs = [
            (False, False, substrate_miller, film_miller),
            (True, False, self.neg_miller(substrate_miller), film_miller),
            (False, True, substrate_miller, self.neg_miller(film_miller)),
            (True, True, self.neg_miller(substrate_miller), self.neg_miller(film_miller)),
        ]

        for i, (itf, area) in enumerate(itfs):
            for j, (flip_sub, flip_film, sub_m, film_m) in enumerate(flip_specs):
                itf_use = itf if (not flip_sub and not flip_film) else self.flip_interface(itf, flip_sub=flip_sub, flip_film=flip_film)
                sub_atoms = AseAtomsAdaptor.get_atoms(itf_use.substrate)
                film_atoms = AseAtomsAdaptor.get_atoms(itf_use.film)
                n_sub_layers = len(GraphEncoder.cluster_layers_by_z(sub_atoms))
                n_film_layers = len(GraphEncoder.cluster_layers_by_z(film_atoms))
                sl = int(self.interface_params.substrate_layers)
                fl = int(self.interface_params.film_layers)
                sub_ratio = n_sub_layers / float(sl)
                film_ratio = n_film_layers / float(fl)
                sub_period_layers = int(round(sub_ratio))
                film_period_layers = int(round(film_ratio))
                # strict stacking consistency check (no rumpling allowed
                if abs(sub_ratio - sub_period_layers) > 1e-6:
                    raise RuntimeError(
                        f"Substrate stacking inconsistency detected: "
                        f"n_sub_layers={n_sub_layers}, sub_layers={sl}, "
                        f"ratio={sub_ratio:.6f} (non-integer). "
                        f"Rumbling or structural distortion not supported."
                    )
                if abs(film_ratio - film_period_layers) > 1e-6:
                    raise RuntimeError(
                        f"Film stacking inconsistency detected: "
                        f"n_film_layers={n_film_layers}, film_layers={fl}, "
                        f"ratio={film_ratio:.6f} (non-integer). "
                        f"Rumbling or structural distortion not supported."
                    )
                records.append({
                    "substrate_miller": [int(x) for x in sub_m],
                    "film_miller": [int(x) for x in film_m],
                    "termination": [str(term[0]), str(term[1])],
                    "cand_id": int(i),
                    "interface": itf_use,
                    "area": float(area),
                    "substrate_bulk": None,
                    "film_bulk": None,
                    "sub_period_layers": int(sub_period_layers),
                    "film_period_layers": int(film_period_layers),
                })

        if build_bulk_refs:
            fl0 = self.interface_params.film_layers
            sl0 = self.interface_params.substrate_layers
            itf_refs = self.collect_candidates(cib, term, film_layers=(fl0+sl0), substrate_layers=(fl0+sl0))
            if len(itf_refs) != len(itfs):
                raise RuntimeError(f"Candidate count mismatch: itfs={len(itfs)}, itf_refs={len(itf_refs)}")
            # Attach bulk references (centered) for each candidate.
            for i, (itf_ref, area_ref) in enumerate(itf_refs):
                for j, (flip_sub, flip_film, sub_m, film_m) in enumerate(flip_specs):
                    itf_ref_use = itf_ref if (not flip_sub and not flip_film) else self.flip_interface(itf_ref, flip_sub=flip_sub, flip_film=flip_film)
                    sub_bulk_i = self.recenter_slab(itf_ref_use.substrate)
                    film_bulk_i = self.recenter_slab(itf_ref_use.film)
                    idx = i * 4 + j
                    records[idx]["substrate_bulk"] = sub_bulk_i
                    records[idx]["film_bulk"] = film_bulk_i

        # Optional structure dump for debugging
        if structure_check:
            out_dir = "results"
            os.makedirs(out_dir, exist_ok=True)
            for rec in records:
                s = rec["substrate_miller"]
                f = rec["film_miller"]
                f_t, s_t = rec["termination"]
                sub_tag = f"{s[0]}{s[1]}{s[2]}"
                film_tag = f"{f[0]}{f[1]}{f[2]}"
                f_t = f_t.replace("/", "-")
                s_t = s_t.replace("/", "-")
                term_tag = f"{s_t}_{f_t}"
                i = rec["cand_id"]
                a = rec["area"]
                stem = f"sub{sub_tag}_film{film_tag}_term{term_tag}_cand{i}"
                # CIFs for visualization
                itf_path = os.path.join(out_dir, f"{stem}_area{round(a)}.cif")
                rec["interface"].to(filename=itf_path)
                # JSON files for runnable artifact
                meta = dict(rec)
                meta["interface"] = rec["interface"].as_dict()
                meta["substrate_bulk"] = rec["substrate_bulk"].as_dict() if rec["substrate_bulk"] is not None else None
                meta["film_bulk"] = rec["film_bulk"].as_dict() if rec["film_bulk"] is not None else None
                json_path = os.path.join(out_dir, f"{stem}_record.json")
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2, cls=MontyEncoder)

        return records

    def sum_interface_records(self, build_bulk_refs=False, structure_check=False):
        """
        Enumerate all (substrate_miller, film_miller, termination) combinations and aggregate candidate records.

        Returns
        -------
        list[dict]: Concatenated records from 'get_interface_records'
        """
        records = []

        for s_idx in self.s_indices:
            for f_idx in self.f_indices:
                cib = self.build_cib(substrate_miller=s_idx, film_miller=f_idx)
                if cib is None or not cib.terminations:
                    continue
                for term in cib.terminations:
                    recs = self.get_interface_records(
                        s_idx, f_idx, term,
                        build_bulk_refs=build_bulk_refs,
                        structure_check=structure_check,
                        cib=cib,
                    )
                    if not recs:
                        continue
                    records.extend(recs)

        return records







