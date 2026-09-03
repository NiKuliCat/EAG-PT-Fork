import math
import pathlib
import sys
import unittest


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from libraries.pbr_reference import (  # noqa: E402
    build_emitter_cdf,
    cosine_hemisphere_pdf,
    disney_f0_kd,
    disney_mixture_pdf,
    eval_disney_brdf,
    ggx_reflection_pdf,
    power_heuristic,
)


class PbrReferenceTests(unittest.TestCase):
    def test_metallic_endpoints(self):
        basecolor = (0.8, 0.3, 0.1)
        self.assertEqual(disney_f0_kd(basecolor, 0.0), ((0.04,) * 3, basecolor))
        self.assertEqual(disney_f0_kd(basecolor, 1.0), (basecolor, (0.0,) * 3))

    def test_roughness_boundaries_are_finite_and_nonnegative(self):
        for roughness in (0.0, 0.02, 1.0):
            value = eval_disney_brdf(
                (0.0, 0.0, 1.0),
                (0.3, 0.0, math.sqrt(0.91)),
                (0.0, 0.0, 1.0),
                (0.7, 0.5, 0.2),
                0.4,
                roughness,
            )
            self.assertTrue(all(math.isfinite(channel) for channel in value))
            self.assertTrue(all(channel >= 0.0 for channel in value))

    def test_pdf_integrals(self):
        normal = (0.0, 0.0, 1.0)
        wo = normal
        basecolor = (0.6, 0.4, 0.2)
        # NDF sampling can reflect a tail below the surface. At low roughness
        # that rejected probability is small, so the valid-hemisphere integral
        # is the useful normalization regression for this sampler.
        roughness = 0.2
        steps_theta = 180
        steps_phi = 360
        integrals = [0.0, 0.0, 0.0]
        for theta_i in range(steps_theta):
            theta = (theta_i + 0.5) * (0.5 * math.pi / steps_theta)
            sin_theta = math.sin(theta)
            cos_theta = math.cos(theta)
            domega = sin_theta * (0.5 * math.pi / steps_theta) * (
                2.0 * math.pi / steps_phi
            )
            for phi_i in range(steps_phi):
                phi = (phi_i + 0.5) * (2.0 * math.pi / steps_phi)
                wi = (sin_theta * math.cos(phi), sin_theta * math.sin(phi), cos_theta)
                integrals[0] += cosine_hemisphere_pdf(normal, wi) * domega
                integrals[1] += ggx_reflection_pdf(normal, wi, wo, roughness) * domega
                integrals[2] += disney_mixture_pdf(
                    normal, wi, wo, basecolor, 0.2, roughness
                ) * domega
        self.assertAlmostEqual(integrals[0], 1.0, delta=2.0e-3)
        self.assertAlmostEqual(integrals[1], 1.0, delta=5.0e-3)
        self.assertAlmostEqual(integrals[2], 1.0, delta=5.0e-3)

    def test_power_heuristic_range(self):
        for a, b in ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.2, 0.7)):
            value = power_heuristic(a, b)
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_hemisphere_energy_bound(self):
        normal = (0.0, 0.0, 1.0)
        wo = normal
        steps_theta = 160
        steps_phi = 320
        energy = [0.0, 0.0, 0.0]
        for theta_i in range(steps_theta):
            theta = (theta_i + 0.5) * (0.5 * math.pi / steps_theta)
            sin_theta = math.sin(theta)
            cos_theta = math.cos(theta)
            domega = sin_theta * (0.5 * math.pi / steps_theta) * (
                2.0 * math.pi / steps_phi
            )
            for phi_i in range(steps_phi):
                phi = (phi_i + 0.5) * (2.0 * math.pi / steps_phi)
                wi = (sin_theta * math.cos(phi), sin_theta * math.sin(phi), cos_theta)
                brdf = eval_disney_brdf(
                    normal, wi, wo, (0.8, 0.7, 0.6), 0.0, 0.4
                )
                for channel in range(3):
                    energy[channel] += brdf[channel] * cos_theta * domega
        self.assertLessEqual(max(energy), 1.05)

    def test_emitter_cdf_and_empty_distribution(self):
        pdf, cdf = build_emitter_cdf([1.0, 2.0, 3.0])
        self.assertAlmostEqual(sum(pdf), 1.0)
        self.assertTrue(all(a <= b for a, b in zip(cdf, cdf[1:])))
        self.assertEqual(cdf[-1], 1.0)
        self.assertEqual(build_emitter_cdf([0.0, -1.0]), ([], []))


if __name__ == "__main__":
    unittest.main()
