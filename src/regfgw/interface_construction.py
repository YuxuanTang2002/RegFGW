import os
import re
import numpy as np
from dataclasses import dataclass
from ase.data import atomic_numbers, vdw_radii
from pymatgen.core import Structure
from pymatgen.core import Lattice
from pymatgen.core.surface import get_symmetrically_distinct_miller_indices
from pymatgen.analysis.interfaces.zsl import ZSLGenerator, vec_area
from pymatgen.analysis.interfaces.coherent_interfaces import CoherentInterfaceBuilder
from pymatgen.io.ase import AseAtomsAdaptor
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
    max_area_ratio_tol: float = 0.09
    max_length_tol: float = 0.03
    max_angle_tol: float = 0.01

@dataclass(frozen=True)
class InterfaceParams:
    """
    Geometric parameters for building coherent interfaces

    Attributes
    ----------
    film_layers, substrate_layers: Slab thickness controls. One layer corresponds to the minimum stacking repeat period.
    gap: float or None. If None, interfacial gap is the sum the maximum vdw radii of the atoms on both sides.
    vacuum: Vacuum thickness above film slab(Å)
    """
    film_layers: int = 3
    substrate_layers: int = 3
    gap: None | float = None
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
        self.substrate = substrate
        self.film = film
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

    # -----------------------------------------------------------------------------
    # Gap determination
    # -----------------------------------------------------------------------------

    @staticmethod
    def gap_from_term(term):
        """
        Compute a conservative interfacial gap (Å) from a (film_term, sub_term) termination pair.
        gap = max_r(film) + max_r(substrate), where radii are ASE vdw radii.
        """
        def max_r_from_term(t):
            symbol = str(t).split("_", 1)[0]
            elements = []

            for e in re.findall(r"[A-Z][a-z]?", symbol):
                if e in atomic_numbers:
                    elements.append(e)

            if not elements:
                raise RuntimeError(f"Failed to parse elements from termination {t}")

            radii = []

            for e in elements:
                r = float(vdw_radii[atomic_numbers[e]])
                if r is None or r < 1e-6:
                    raise RuntimeError(f"ASE vdw radius not available for element {e}")
                radii.append(r)

            return max(radii)

        f_term, s_term = term
        r_f = max_r_from_term(f_term)
        r_s = max_r_from_term(s_term)
        gap = r_f + r_s

        return gap

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

        if gap is None:
            gap = self.gap_from_term(term)

        interfaces = cib.get_interfaces(
            term,
            gap=gap,
            vacuum_over_film=self.interface_params.vacuum,
            substrate_thickness=substrate_layers,
            film_thickness=film_layers,)

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
    # Thickness matching heuristics to build bulk references
    # -------------------------------------------------------------------------

    def find_best_film_layers(self, cib: CoherentInterfaceBuilder, term, t_target: float,
                             sub_layers: int, film_layers_start: int, film_layers_stop: int):
        """
        Choose film slab thickness (layers) such that film z-thickness ~ t_target.

        Notes
        -----
        The first candidate in each layer setting is used to estimate thickness.
        """
        best_fl = film_layers_start
        best_err = None

        for fl in range(film_layers_start, film_layers_stop+1):
            candidates = self.collect_candidates(cib, term, film_layers=fl, substrate_layers=sub_layers)

            if not candidates:
                continue

            itf = candidates[0][0]
            t_film = self.extract_thickness(itf.film)
            err = abs(t_film - t_target)
            if best_err is None or err < best_err:
                best_err = err
                best_fl = fl

        return best_fl

    def find_best_sub_layers(self, cib: CoherentInterfaceBuilder, term, t_target: float,
                             film_layers: int, sub_layers_start: int, sub_layers_stop: int):
        """Choose substrate slab thickness (layers) such that substrate z-thickness ~ t_target."""
        best_sl = sub_layers_start
        best_err = None

        for sl in range(sub_layers_start, sub_layers_stop+1):
            candidates = self.collect_candidates(cib, term, film_layers=film_layers, substrate_layers=sl)

            if not candidates:
                continue

            itf = candidates[0][0]
            t_sub = self.extract_thickness(itf.substrate)
            err = abs(t_sub - t_target)
            if best_err is None or err < best_err:
                best_err = err
                best_sl = sl

        return best_sl

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def get_interface_records(
            self, substrate_miller, film_miller, term,
            build_bulk_refs=True, structure_check=False,
    ):
        """
        Build full candidate records for a given (substrate_miller, film_miller, termination).

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
        cib = self.build_cib(substrate_miller=substrate_miller, film_miller=film_miller)

        if cib is None:
            return None

        itfs = self.collect_candidates(cib, term)

        if not itfs:
            return None

        gap = self.interface_params.gap

        if gap is None:
            gap = self.gap_from_term(term)

        records = []

        for i, (itf, area) in enumerate(itfs):
            sub_atoms = AseAtomsAdaptor.get_atoms(itf.substrate)
            film_atoms = AseAtomsAdaptor.get_atoms(itf.film)
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
                "substrate_miller": substrate_miller,
                "film_miller": film_miller,
                "termination": term,
                "cand_id": i,
                "interface": itf,
                "gap":  gap,
                "area": area,
                "substrate_bulk": None,
                "film_bulk": None,
                "sub_period_layers": sub_period_layers,
                "film_period_layers": film_period_layers,
            })

        if build_bulk_refs:
            # Use the first candidate as a representative to set a target thickness.
            itf0 = itfs[0][0]
            t_target = self.extract_thickness(itf0)
            fl0 = self.interface_params.film_layers
            sl0 = self.interface_params.substrate_layers
            layers_search_stop = 3 * (fl0 + sl0)
            best_fl = self.find_best_film_layers(
                cib, term, t_target,
                sub_layers=sl0, film_layers_start=fl0, film_layers_stop=layers_search_stop,
            )
            best_sl = self.find_best_sub_layers(
                cib, term, t_target,
                film_layers=fl0, sub_layers_start=sl0, sub_layers_stop=layers_search_stop,
            )
            itf_refs = self.collect_candidates(cib, term, film_layers=best_fl, substrate_layers=best_sl)
            # Expect the same number of candidates if only changing the layer thickness.
            if len(itf_refs) != len(itfs):
                raise RuntimeError(f"Candidate count mismatch: itfs={len(itfs)}, itf_refs={len(itf_refs)}")
            # Attach bulk references (centered) for each candidate.
            for i, (itf_ref, area_ref) in enumerate(itf_refs):
                sub_bulk_i = self.recenter_slab(itf_ref.substrate)
                film_bulk_i = self.recenter_slab(itf_ref.film)
                records[i]["substrate_bulk"] = sub_bulk_i
                records[i]["film_bulk"] = film_bulk_i

        # Optional structure dump for debugging
        if structure_check:
            out_dir = "results"
            os.makedirs(out_dir, exist_ok=True)
            s = records[0]["substrate_miller"]
            f = records[0]["film_miller"]
            f_t, s_t = records[0]["termination"]
            g = records[0]["gap"]
            sub_tag = f"{s[0]}{s[1]}{s[2]}"
            film_tag = f"{f[0]}{f[1]}{f[2]}"
            f_t = f_t.replace("/", "-")
            s_t = s_t.replace("/", "-")
            term_tag = f"{s_t}_{f_t}"
            gap_tag = f"{g:.2f}".replace(".", "p")
            for rec in records:
                i = rec["cand_id"]
                a = rec["area"]
                itf_fname = (f"sub{sub_tag}_film{film_tag}_"
                             f"term{term_tag}_cand{i}_gap{gap_tag}_area{round(a)}.cif")
                itf_path = os.path.join(out_dir, itf_fname)
                rec["interface"].to(filename=itf_path)
                if rec["substrate_bulk"] is not None:
                    sub_fname = (f"sub{sub_tag}_film{film_tag}_"
                                 f"term{term_tag}_cand{i}_sub_bulk.cif")
                    sub_path = os.path.join(out_dir, sub_fname)
                    rec["substrate_bulk"].to(filename=sub_path)
                if rec["film_bulk"] is not None:
                    film_fname = (f"sub{sub_tag}_film{film_tag}_"
                                  f"term{term_tag}_cand{i}_film_bulk.cif")
                    film_path = os.path.join(out_dir, film_fname)
                    rec["film_bulk"].to(filename=film_path)

        return records

    def sum_interface_records(self, build_bulk_refs=True, structure_check=False):
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
                    recs = self.get_interface_records(s_idx, f_idx, term, build_bulk_refs=build_bulk_refs, structure_check=structure_check)
                    if not recs:
                        continue
                    records.extend(recs)

        return records







