from itertools import combinations_with_replacement
import string

def generate_fully_connected_gmd_contractions(Lmax=3, Kmax=3):
    """
    Generate fully connected δ-only contractions for moments[1..Lmax] up to multiplicity Kmax.
    Returns a dict: {ells_canon: [(inputs_tuple, output_subscript)]}
    """
    contractions = {}

    # Include 2-body scalar from moments[0]
    contractions[(0,)] = [(("ar",), "ar")]

    radial_letters = list(string.ascii_lowercase[17:])  # r,s,t,...
    ang_letters_pool = list(string.ascii_lowercase[8:])  # i,j,k,...

    seen_multisets = set()

    # --- Helper: recursive backtracking for cross-tensor pairings ---
    def find_pairings(slots, current_pairs, tensor_graph):
        """
        slots: list of remaining angular slots (tensor_index, slot_index)
        current_pairs: list of assigned pairs
        tensor_graph: dict tensor_index -> set of connected tensor indices
        Returns list of valid fully connected pairings
        """
        if not slots:
            # check connectivity
            tensors = list(tensor_graph.keys())
            if len(tensors) <= 1:
                return [list(current_pairs)]
            # simple DFS for connectivity
            visited = set()
            stack = [tensors[0]]
            while stack:
                t = stack.pop()
                if t in visited:
                    continue
                visited.add(t)
                stack.extend(tensor_graph.get(t, []))
            if len(visited) == len(tensors):
                return [list(current_pairs)]
            return []

        first = slots[0]
        pairings_found = []
        for j in range(1, len(slots)):
            second = slots[j]
            if first[0] == second[0]:
                continue  # cross-tensor only
            # update tensor graph
            tg = {k: set(v) for k,v in tensor_graph.items()}
            tg.setdefault(first[0], set()).add(second[0])
            tg.setdefault(second[0], set()).add(first[0])
            # recurse
            rest = slots[1:j] + slots[j+1:]
            pairings_found.extend(find_pairings(rest, current_pairs + [(first, second)], tg))
        return pairings_found

    # --- Main loop: generate candidate multisets ---
    for k in range(2, Kmax+1):
        for ells in combinations_with_replacement(range(1, Lmax+1), k):  # only ell>=1
            ells_canon = tuple(sorted(ells, reverse=True))
            if ells_canon in seen_multisets:
                continue
            seen_multisets.add(ells_canon)

            total_slots = sum(ells_canon)
            if total_slots % 2 != 0:
                continue  # cannot pair fully

            # build flat angular slots
            slots = []
            for t, ell in enumerate(ells_canon):
                slots.extend([(t, i) for i in range(ell)])

            # find at least one fully connected pairing
            pairings = find_pairings(slots, [], {})
            if not pairings:
                continue  # invalid multiset

            chosen_pairing = pairings[0]  # pick first canonical pairing

            # assign angular letters
            ang_letter_map = {}
            for idx, ((t0,p0),(t1,p1)) in enumerate(chosen_pairing):
                letter = ang_letters_pool[idx]
                ang_letter_map[(t0,p0)] = letter
                ang_letter_map[(t1,p1)] = letter

            # build einsum input subscripts
            inputs = []
            for tidx, ell in enumerate(ells_canon):
                R = radial_letters[tidx]
                angs = "".join(ang_letter_map[(tidx, s)] for s in range(ell))
                sub = "a" + R + angs
                inputs.append(sub)

            output = "a" + "".join(radial_letters[:len(ells_canon)])

            einsum_str = ", ".join(inputs) + " -> " + output
            contractions[ells_canon] = einsum_str

    return contractions

# --- Demo ---
if __name__ == "__main__":
    contrs = generate_fully_connected_gmd_contractions(Lmax=2, Kmax=3)
    for ranks, specs in sorted(contrs.items()):
        print(ranks, "->", specs)
