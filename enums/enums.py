from enum import Enum


class MapName(Enum):
    small = 0  # 4x4
    medium = 1  # 8x8
    large = 2  # 16x16
    huge = 3  # 32x32
    colossal = 4  # 100x100


class PolicyType(Enum):
    optimal = 0
    stochastic = 1


class ActionSet(Enum):
    default = 0
    slippery = 1


class EnvType(Enum):
    default = 'default-v0'
    fall = 'fall-v0'
