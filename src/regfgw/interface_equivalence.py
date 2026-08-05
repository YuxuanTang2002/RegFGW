from dataclasses import dataclass
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.interface import Interface
from tqdm import tqdm
from typing import List, Tuple

# -----------------------------------------------------------------------------
# Parameter containers
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class InterfaceMatchParams:
    """
    Parameters controlling periodic structure matching.

    Attributes
    ----------
    ltol: fractional lattice-length tolerance
    stol: normalized site-position tolerance
    angle_tol: lattice-angle tolerance (degrees)
    """
    ltol: float = 1e-5
    stol: float = 1e-3
    angle_tol: float = 1e-3

    def __post_init__(self):
        if self.ltol < 0.0:
            raise ValueError("ltol must be non-negative.")

        if self.stol < 0.0:
            raise ValueError("stol must be non-negative.")

        if self.angle_tol < 0.0:
            raise ValueError("angle_tol must be non-negative.")

@dataclass(frozen=True)
class InterfaceGroup:
    """
    Group of equivalent periodic structures

    Attributes
    ----------
    rep_index: index of the representative structure
    member_indices: indices of all equivalent structures
    """
    rep_index: int
    member_indices: Tuple[int, ...]

# -----------------------------------------------------------------------------
# Interface reduction
# -----------------------------------------------------------------------------

class InterfaceMatcher:
    def __init__(self, params: InterfaceMatchParams):
        self.params = params
        self.matcher = StructureMatcher(
            ltol=self.params.ltol,
            stol=self.params.stol,
            angle_tol=self.params.angle_tol,
            primitive_cell=False,
            scale=False,
            attempt_supercell=False,
            allow_subset=False,
        )

    def check_equivalence(self, interface_a: Interface, interface_b: Interface):
        """Determine whether two interfaces are structurally equivalent."""
        return self.matcher.fit(interface_a, interface_b)

    def group_interfaces(self, interfaces: List[Interface]):
        """
        Partition interface structures into equivalence classes.

        Parameters
        ----------
        interfaces: interface structures to group

        Returns
        -------
        List[InterfaceGroup]: equivalence classes
        """
        groups: List[List[int]] = []

        with tqdm(total=len(interfaces), desc="Grouping equivalent interfaces") as pbar:
            for index, interface in enumerate(interfaces):
                for group in groups:
                    rep = interfaces[group[0]]
                    if self.check_equivalence(interface, rep):
                        group.append(index)
                        break
                else:
                    groups.append([index])
                pbar.set_postfix(groups=len(groups))
                pbar.update(1)

        return [
            InterfaceGroup(rep_index=group[0], member_indices=tuple(group))
            for group in groups
        ]
