from __future__ import annotations

SHAPE_ATTRIBUTE_MAP = {
    0: {
        0: "sleeveless",
        1: "short-sleeve",
        2: "medium-sleeve",
        3: "long-sleeve",
        4: "not long-sleeve",
        5: "NA",
    },
    1: {
        0: "three-point",
        1: "medium short",
        2: "three-quarter",
        3: "long",
        4: "NA",
    },
    2: {
        0: "no socks",
        1: "socks",
        2: "leggings",
        3: "NA",
    },
    3: {
        0: "no hat",
        1: "hat",
        2: "NA",
    },
    4: {
        0: "no glasses",
        1: "eyeglasses",
        2: "sunglasses",
        3: "glasses in hand or on clothes",
        4: "NA",
    },
    5: {
        0: "no neckwear",
        1: "neckwear",
        2: "NA",
    },
    6: {
        0: "no wrist wearing",
        1: "wrist wearing",
        2: "NA",
    },
    7: {
        0: "no ring",
        1: "ring",
        2: "NA",
    },
    8: {
        0: "no waist accessories",
        1: "belt",
        2: "clothing at waist",
        3: "hidden",
        4: "NA",
    },
    9: {
        0: "V-shape neckline",
        1: "square neckline",
        2: "round neckline",
        3: "standing collar",
        4: "lapel",
        5: "suspenders",
        6: "NA",
    },
    10: {
        0: "cardigan",
        1: "not cardigan",
        2: "NA",
    },
    11: {
        0: "navel visible",
        1: "navel covered",
        2: "NA",
    },
}

FABRIC_ATTRIBUTE_MAP = {
    0: "denim",
    1: "cotton",
    2: "leather",
    3: "furry",
    4: "knitted",
    5: "chiffon",
    6: "other",
    7: "NA",
}

PATTERN_ATTRIBUTE_MAP = {
    0: "floral",
    1: "graphic",
    2: "striped",
    3: "pure color",
    4: "lattice",
    5: "other",
    6: "color block",
    7: "NA",
}

FABRIC_PART_NAMES = ["upper_fabric", "lower_fabric", "outer_fabric"]
PATTERN_PART_NAMES = ["upper_pattern", "lower_pattern", "outer_pattern"]

SHAPE_PART_NAMES = [
    "sleeve_length",
    "lower_clothing_length",
    "socks",
    "hat",
    "glasses",
    "neckwear",
    "wrist_wearing",
    "ring",
    "waist_accessories",
    "neckline",
    "cardigan",
    "covering_navel",
]