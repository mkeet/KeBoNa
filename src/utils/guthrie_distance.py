"""
guthrie_distance.py - linguistic distance between two languages from their Guthrie codes.

A code is a zone letter followed by a two-digit number. The tens digit is the group within the
zone and the units digit is the language within the group, so isiZulu is S42, isiXhosa S41 and
Shona S10. The distance between two languages is

    10 * zone distance + 3 * group difference + 1 * language difference

The zone distance comes in two forms. guthrie_graph_distance counts the zone borders crossed
on the map and is the one the experiments use. guthrie_full_distance takes the gap between the
zone letters in the alphabet.

    guthrie_graph_distance("zulu", "xhosa")    -> 1     same group
    guthrie_graph_distance("zulu", "shona")    -> 11    same zone, three groups apart
    guthrie_graph_distance("zulu", "swahili")  -> 30    three zone borders
"""

import re
from collections import deque

# 16 zones, since I, O and Q were never assigned
_ZONE_ORDER = "ABCDEFGHJKLMNPRS"
_ZONE_INDEX = {z: i for i, z in enumerate(_ZONE_ORDER)}

# Zones that share a border on the map (Maho 2009 NUGL).
_ZONE_ADJ = {
    "A": ["B", "C"],
    "B": ["A", "C", "H"],
    "C": ["A", "B", "D", "H", "L"],
    "D": ["C", "J"],
    "E": ["J", "G"],
    "F": ["G", "J", "M"],
    "G": ["E", "F", "P"],
    "H": ["B", "C", "R"],
    "J": ["D", "E", "F"],
    "K": ["L", "R"],
    "L": ["C", "K", "M"],
    "M": ["F", "L", "N", "P"],
    "N": ["M", "P", "S"],
    "P": ["G", "M", "N"],
    "R": ["H", "K", "S"],
    "S": ["N", "R"],
}

ZONE_WEIGHT  = 10
GROUP_WEIGHT = 3
LANG_WEIGHT  = 1


# Keys are the corpus/model folder names, plus common alternative names.
GUTHRIE_CODES = {
    # Nguni (Zone S, group 4x)
    "zulu":     "S42",
    "xhosa":    "S41",
    "ndebele":  "S44",
    "swati":    "S43",
    # Shona (Zone S, group 1x)
    "shona":    "S10",
    # Sotho-Tswana (Zone S, group 3x), written disjunctively
    "sepedi":   "S32",
    "setswana": "S31",
    "sesotho":  "S33",
    "sotho":    "S33",  # same as sesotho
    "tswana":   "S31",  # same as setswana
    # Zones G and K
    "swahili":  "G42",
    "lozi":     "K21",
    "silozi":   "K21",  # same as lozi
    # Other groups of zone S
    "pedi":     "S32",  # same as sepedi
    "venda":    "S21",  # Tshivenda, Venda group S20
    "tshivenda":"S21",  # same as venda
    "tsonga":   "S53",  # Xitsonga, Tswa-Ronga group S50
    "xitsonga": "S53",  # same as tsonga
    # Great Lakes (Maho zones JD and JE)
    "kinyarwanda": "JD61",  # Rwanda
    "kirundi":     "JD62",  # Rundi
    "luganda":     "JE15",  # Ganda
    "ganda":       "JE15",  # same as luganda
    "runyankore":  "JE13",  # Nyankore
    # Zone E (highland East Africa)
    "kikuyu":   "E51",  # Gikuyu
    "gikuyu":   "E51",  # same as kikuyu
    "kamba":    "E55",
    # Zone N (Malawi/Zambia)
    "chichewa": "N31",  # Chewa / Nyanja
    "nyanja":   "N31",  # same as chichewa
    "tumbuka":  "N21",
    # Zone M (Zambia)
    "bemba":    "M42",
    # Zone L (S DRC / C Zambia)
    "luba":     "L31",  # Luba-Kasai / Tshiluba
    "tshiluba": "L31",  # same as luba
    "ciluba":   "L31",  # same as luba
    # Zone P (SE Tanzania / N Mozambique)
    "yao":      "P21",
    "makhuwa":  "P31",
    # Zone B (Gabon coast)
    "lumbu":    "B44",  # Yilumbu (ISO lup)
    "yilumbu":  "B44",  # same as lumbu
    # Zone C (Congo basin)
    "lingala":  "C36",
    # Zone H (Kongo/Angola coast)
    "kongo":    "H16",  # Kikongo
    "kikongo":  "H16",  # same as kongo
    "kimbundu": "H21",
    # Zone R (Angola/Namibia)
    "umbundu":  "R11",
    "herero":   "R31",
    # not a Bantu language, so it has no code and no distance
    "english":  None,
    # NCHLT corpus folders
    "zu":      "S42",  # same as zulu
    "xh":      "S41",  # same as xhosa
    "nd":      "S44",  # same as ndebele
    "sw":      "S43",  # same as swati
}


def parse_code(code: str):
    """
    Splits a code such as S42 into its zone letter and its number. A Maho subzone code such as
    JE15 has a second letter, which is folded into its parent zone J. A code that cannot be
    parsed raises ValueError.
    """
    code = code.strip().upper()
    m = re.fullmatch(r"([A-S])([A-S]?)(\d+)", code)
    if not m:
        raise ValueError(f"Cannot parse Guthrie code: {code!r}")
    return m.group(1), int(m.group(3))


def guthrie_distance(lang_a, lang_b):
    """
    Returns the zone difference, the group difference and the language difference between two
    languages of GUTHRIE_CODES. The zone difference here is the gap between the zone letters
    in the alphabet.
    """
    code_a = GUTHRIE_CODES[lang_a]
    code_b = GUTHRIE_CODES[lang_b]
    zone_a, num_a = parse_code(code_a)
    zone_b, num_b = parse_code(code_b)
    zone_diff  = abs(_ZONE_INDEX[zone_a] - _ZONE_INDEX[zone_b])
    group_diff = abs((num_a // 10) - (num_b // 10))
    lang_diff  = abs((num_a % 10)  - (num_b % 10))
    return zone_diff, group_diff, lang_diff


def guthrie_full_distance(lang_a, lang_b):
    """
    Returns 10 * zone difference + 3 * group difference + 1 * language difference, with the
    zone difference taken as the gap between the zone letters in the alphabet. From isiZulu
    this gives 1 for isiXhosa, 4 for Sesotho, 11 for Shona, 67 for Silozi and 90 for Kiswahili.
    """
    zone_diff, group_diff, lang_diff = guthrie_distance(lang_a, lang_b)
    return ZONE_WEIGHT * zone_diff + GROUP_WEIGHT * group_diff + LANG_WEIGHT * lang_diff


_zone_bfs_cache = {}


def zone_graph_distance(zone_a: str, zone_b: str):
    """
    Returns the smallest number of zone borders between two zones, found by a breadth-first
    search over _ZONE_ADJ. The same zone gives 0.
    """
    za, zb = zone_a.strip().upper(), zone_b.strip().upper()
    if za == zb:
        return 0
    key = (za, zb)
    if key in _zone_bfs_cache:
        return _zone_bfs_cache[key]
    seen = {za}
    q = deque([(za, 0)])
    while q:
        z, d = q.popleft()
        for nb in _ZONE_ADJ.get(z, []):
            if nb == zb:
                _zone_bfs_cache[key] = _zone_bfs_cache[(zb, za)] = d + 1
                return d + 1
            if nb not in seen:
                seen.add(nb)
                q.append((nb, d + 1))
    _zone_bfs_cache[key] = None
    return None


def guthrie_graph_components(lang_a, lang_b):
    """
    Returns the zone borders crossed, the group difference and the language difference
    between two languages.
    """
    za, na = parse_code(GUTHRIE_CODES[lang_a])
    zb, nb = parse_code(GUTHRIE_CODES[lang_b])
    return (zone_graph_distance(za, zb),
            abs((na // 10) - (nb // 10)),
            abs((na % 10) - (nb % 10)))


def guthrie_graph_distance(lang_a, lang_b,
                           w_zone=ZONE_WEIGHT, w_group=GROUP_WEIGHT, w_lang=LANG_WEIGHT):
    """
    Returns the weighted distance, with the zone term taken as the number of zone borders
    crossed. The weights default to 10, 3 and 1. From isiZulu this gives 1 for isiXhosa,
    4 for Sesotho, 11 for Shona, 27 for Silozi and 30 for Kiswahili.
    """
    hops, gd, ld = guthrie_graph_components(lang_a, lang_b)
    if hops is None:
        return None
    return w_zone * hops + w_group * gd + w_lang * ld


def guthrie_graph_lexrank(lang_a, lang_b):
    """
    Returns an ordering with no weights to choose. Zone borders dominate, then the group
    difference, then the language difference. The three are packed into one sortable number,
    100 * borders + 10 * group + language, for use in a rank correlation.
    """
    hops, gd, ld = guthrie_graph_components(lang_a, lang_b)
    if hops is None:
        return None
    return 100 * hops + 10 * gd + ld


if __name__ == "__main__":
    source = "zulu"
    print(f"\nGuthrie distances from {source} ({GUTHRIE_CODES[source]})\n")
    print(f"  {'language':<14}{'code':<8}{'distance':>9}  ({ZONE_WEIGHT} x zone borders + "
          f"{GROUP_WEIGHT} x group + {LANG_WEIGHT} x language)")
    print("  " + "-" * 62)
    for lang, code in GUTHRIE_CODES.items():
        if code is None:
            print(f"  {lang:<14}{'-':<8}{'-':>9}  no Guthrie code")
            continue
        hops, gd, ld = guthrie_graph_components(source, lang)
        print(f"  {lang:<14}{code:<8}{guthrie_graph_distance(source, lang):>9}  "
              f"({ZONE_WEIGHT}x{hops} + {GROUP_WEIGHT}x{gd} + {LANG_WEIGHT}x{ld})")
    print()
