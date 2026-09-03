"""CPU reference math for the EAG-PT Disney metallic-roughness renderer."""

from __future__ import annotations

import bisect
import math
from collections.abc import Sequence


Vec3 = tuple[float, float, float]
PI = math.pi
EPSILON = 1.0e-8
MIN_ROUGHNESS = 0.02
TRUNCATION_SIGMA = 3.0
TRUNCATED_GAUSSIAN_Z = 1.0 - math.exp(-0.5 * TRUNCATION_SIGMA**2)


def _dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _normalize(v: Vec3) -> Vec3:
    length = math.sqrt(max(_dot(v, v), EPSILON))
    return (v[0] / length, v[1] / length, v[2] / length)


def _add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def luminance(rgb: Vec3) -> float:
    return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]


def disney_f0_kd(basecolor: Vec3, metallic: float) -> tuple[Vec3, Vec3]:
    metallic = min(max(metallic, 0.0), 1.0)
    f0 = tuple(0.04 * (1.0 - metallic) + c * metallic for c in basecolor)
    kd = tuple((1.0 - metallic) * c for c in basecolor)
    return f0, kd


def fresnel_schlick(cos_theta: float, f0: Vec3) -> Vec3:
    factor = max(0.0, 1.0 - cos_theta) ** 5
    return tuple(c + (1.0 - c) * factor for c in f0)


def ggx_distribution(no_h: float, roughness: float) -> float:
    roughness = min(max(roughness, MIN_ROUGHNESS), 1.0)
    alpha = roughness * roughness
    alpha2 = alpha * alpha
    denominator = no_h * no_h * (alpha2 - 1.0) + 1.0
    return alpha2 / max(PI * denominator * denominator, EPSILON)


def smith_g1(no_v: float, roughness: float) -> float:
    roughness = min(max(roughness, MIN_ROUGHNESS), 1.0)
    alpha = roughness * roughness
    k = 0.5 * alpha
    return no_v / max(no_v * (1.0 - k) + k, EPSILON)


def eval_disney_brdf(
    normal: Vec3,
    wi: Vec3,
    wo: Vec3,
    basecolor: Vec3,
    metallic: float,
    roughness: float,
) -> Vec3:
    normal = _normalize(normal)
    wi = _normalize(wi)
    wo = _normalize(wo)
    no_l = max(_dot(normal, wi), 0.0)
    no_v = max(_dot(normal, wo), 0.0)
    if no_l <= 0.0 or no_v <= 0.0:
        return (0.0, 0.0, 0.0)

    half_vector = _normalize(_add(wi, wo))
    no_h = max(_dot(normal, half_vector), 0.0)
    vo_h = max(_dot(wo, half_vector), 0.0)
    f0, kd = disney_f0_kd(basecolor, metallic)
    fresnel = fresnel_schlick(vo_h, f0)
    distribution = ggx_distribution(no_h, roughness)
    geometry = smith_g1(no_l, roughness) * smith_g1(no_v, roughness)
    specular_scale = distribution * geometry / max(4.0 * no_l * no_v, EPSILON)
    return tuple(kd[i] / PI + fresnel[i] * specular_scale for i in range(3))


def disney_specular_probability(basecolor: Vec3, metallic: float) -> float:
    f0, kd = disney_f0_kd(basecolor, metallic)
    denominator = luminance(f0) + luminance(kd)
    probability = luminance(f0) / denominator if denominator > EPSILON else 0.5
    return min(max(probability, 0.1), 0.9)


def cosine_hemisphere_pdf(normal: Vec3, wi: Vec3) -> float:
    return max(_dot(_normalize(normal), _normalize(wi)), 0.0) / PI


def ggx_reflection_pdf(
    normal: Vec3, wi: Vec3, wo: Vec3, roughness: float
) -> float:
    normal = _normalize(normal)
    wi = _normalize(wi)
    wo = _normalize(wo)
    if _dot(normal, wi) <= 0.0 or _dot(normal, wo) <= 0.0:
        return 0.0
    half_vector = _normalize(_add(wi, wo))
    no_h = max(_dot(normal, half_vector), 0.0)
    vo_h = max(_dot(wo, half_vector), 0.0)
    if vo_h <= 0.0:
        return 0.0
    return ggx_distribution(no_h, roughness) * no_h / max(4.0 * vo_h, EPSILON)


def disney_mixture_pdf(
    normal: Vec3,
    wi: Vec3,
    wo: Vec3,
    basecolor: Vec3,
    metallic: float,
    roughness: float,
) -> float:
    p_spec = disney_specular_probability(basecolor, metallic)
    return (1.0 - p_spec) * cosine_hemisphere_pdf(
        normal, wi
    ) + p_spec * ggx_reflection_pdf(normal, wi, wo, roughness)


def power_heuristic(pdf_a: float, pdf_b: float) -> float:
    a2 = max(pdf_a, 0.0) ** 2
    b2 = max(pdf_b, 0.0) ** 2
    return a2 / (a2 + b2) if a2 + b2 > 0.0 else 0.0


def truncated_gaussian_area_pdf(
    u: float, v: float, scale_x: float, scale_y: float
) -> float:
    radius2 = u * u + v * v
    if radius2 > TRUNCATION_SIGMA**2 or scale_x <= 0.0 or scale_y <= 0.0:
        return 0.0
    normalization = 2.0 * PI * scale_x * scale_y * TRUNCATED_GAUSSIAN_Z
    return math.exp(-0.5 * radius2) / normalization


def build_emitter_cdf(powers: Sequence[float]) -> tuple[list[float], list[float]]:
    positive = [max(float(power), 0.0) for power in powers]
    total = sum(positive)
    if total <= 0.0:
        return [], []
    pdf = [power / total for power in positive]
    cdf: list[float] = []
    cumulative = 0.0
    for probability in pdf:
        cumulative += probability
        cdf.append(cumulative)
    cdf[-1] = 1.0
    return pdf, cdf


def sample_cdf(cdf: Sequence[float], sample: float) -> int:
    if not cdf:
        raise ValueError("Cannot sample an empty emitter CDF")
    return min(bisect.bisect_left(cdf, min(max(sample, 0.0), 1.0)), len(cdf) - 1)
