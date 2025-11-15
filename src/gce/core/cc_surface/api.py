try:
    from cc.core.api import RunBundle, Verdict, compute_verdict  # real package
except Exception:
    from .validators import RunBundle, Verdict
    from .composition import compute_cc, classify_cc
    from .recommend import make_recommendation, make_next_tests, make_checklist

    def compute_verdict(bundle: RunBundle) -> Verdict:
        CC = compute_cc(bundle)
        label = classify_cc(CC)
        rec = make_recommendation(bundle, CC, label)
        nxt = make_next_tests(bundle, CC, label)
        chk = make_checklist(bundle)
        return Verdict(CC=CC, label=label, recommendation=rec, next_tests=nxt, checklist=chk)
