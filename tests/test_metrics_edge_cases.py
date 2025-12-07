from gce.core.cc_surface.metrics import compute_cc_max  # or whatever your API is

def test_cc_max_handles_zero_denom():
    j_dfa = 0.0
    j_dp = 0.0
    j_obs = 0.0  # or very small ~1e-4

    cc_max = compute_cc_max(j_obs=j_obs, j_dfa=j_dfa, j_dp=j_dp)
    assert cc_max == pytest.approx(1.0)
