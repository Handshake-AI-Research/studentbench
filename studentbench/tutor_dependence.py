"""Table 7: AI–human equivalence allowing students to share a human tutor.

CR2 uses the identity working covariance and coefficient-specific Satterthwaite
degrees of freedom. AI sessions are singleton clusters; human sessions share
one cluster per assigned tutor. This is not Welch inference on tutor means.
"""

from pathlib import Path
import numpy as np
from scipy import stats
from .engagement import digest, write_json
from .data import load_sessions
from .journal import Journal


def cr2(gains, human, groups):
    """Return the AI-minus-human contrast, CR2 SE, and Satterthwaite df.

    For contrast c, p_g = M[:,g] (I-H_gg)^(-1/2) X_g (X'X)^(-1)c.
    Variance is sum_g (p_g'y)^2; df = tr(P'P)^2 / tr((P'P)^2).
    These direct equations make the small-cluster correction inspectable.
    """
    y, human, groups = (
        np.asarray(gains, float),
        np.asarray(human, float),
        np.asarray(groups),
    )
    X = np.column_stack([np.ones(len(y)), human])
    bread = np.linalg.inv(X.T @ X)
    beta = bread @ X.T @ y
    residual = y - X @ beta
    columns = []
    scores = []
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        Xg = X[indices]
        eigenvalues, eigenvectors = np.linalg.eigh(
            np.eye(len(indices)) - Xg @ bread @ Xg.T
        )
        assert eigenvalues.min() > 1e-10
        adjustment = (eigenvectors / np.sqrt(eigenvalues)) @ eigenvectors.T
        direction = adjustment @ Xg @ bread[:, 1]
        influence = -X @ bread @ Xg.T @ direction
        influence[indices] += direction
        columns.append(influence)
        scores.append(float(direction @ residual[indices]))
    P = np.column_stack(columns)
    gram = P.T @ P
    variance = float(np.sum(np.square(scores)))
    degrees = float(np.trace(gram) ** 2 / np.sum(gram**2))
    return -float(beta[1]), float(np.sqrt(variance)), degrees


def run(data_dir, output_dir):
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sessions = load_sessions(data_dir, output_dir / "sessions")
    rows = sessions.loc[sessions.kind.isin(["ai", "human"])].to_dict("records")
    for row in rows:
        row["human"] = row["kind"] == "human"
        row["tutor"] = row["human_tutor_id"]
        row["gain"] = row["gain_pp"]
    assert len(rows) == 2279
    results = []
    journal = Journal(output_dir / "sections.jsonl")
    signature = digest(__file__) + digest(output_dir / "sessions/participant_rows.csv")
    for section in ["quant", "verbal"]:
        cached = journal.get(section, signature)
        if cached is not None:
            results.append(cached)
            continue
        selected = [r for r in rows if r["section"] == section]
        y = np.array([r["gain"] for r in selected])
        human = np.array([r["human"] for r in selected])
        groups = [
            str(r["tutor"]) if r["human"] else "ai:" + r["student_id"] for r in selected
        ]
        assert len({r["tutor"] for r in selected if r["human"]}) == 9
        estimate, se, df = cr2(y, human, groups)
        ai, hu = y[~human], y[human]
        margin = 0.25 * np.sqrt(
            ((len(ai) - 1) * ai.var(ddof=1) + (len(hu) - 1) * hu.var(ddof=1))
            / (len(y) - 2)
        )
        interval = estimate + np.array([-1, 1]) * stats.t.ppf(0.95, df) * se
        p = max(
            stats.t.sf((estimate + margin) / se, df),
            stats.t.cdf((estimate - margin) / se, df),
        )
        result = dict(
            section=section,
            n_ai=len(ai),
            n_human=len(hu),
            human_tutors=9,
            estimate=float(estimate),
            se=float(se),
            df=float(df),
            margin=float(margin),
            ci90=interval.tolist(),
            p=float(p),
        )
        results.append(result)
        journal.save(section, signature, result)
    write_json(
        output_dir / "results.json",
        dict(complete=True, status="complete", results=results),
    )
    return results
