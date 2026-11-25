"""Evaluate the post-pilot, source-aware false-positive threshold without opening new contest images."""

import json
from pathlib import Path
import numpy as np

from .external_pilot import axis, binary, FLOOD, ROOT
from .metrics import CODES, evaluate
from .prefix_pilot import filtered_training
from .train_head import decisions, fit, probabilities
from .weight_pilot import pilot_data


def wilson(n,d):
    z=1.95996398454
    p=n/d
    center=(p+z*z/(2*d))/(1+z*z/d)
    spread=z*np.sqrt(p*(1-p)/d+z*z/(4*d*d))/(1+z*z/d)
    return [center-spread,center+spread]


def run():
    with np.load(ROOT/"floodnet/pilot-features.npz",allow_pickle=False) as a:
        x,y,where=a["features"],a["truth"],a["split"]
    xtrain,ytrain,xtest,ytest=x[where==0],y[where==0],x[where==1],y[where==1]
    projection=axis(xtrain,ytrain)
    ext=xtest@projection
    base=binary(ytest,ext>=0)
    rng=np.random.default_rng(173)
    null=np.array([binary(ytest,xtest@axis(xtrain,rng.permutation(ytrain))>=0)["f1"] for _ in range(1000)])
    ref=json.loads(Path("results/prefix_pilot_fpr20.json").read_text())
    groups,feat,gold,linked=pilot_data(Path("results/train_audit_full.json"),Path("data/prefix-pilot/exposure.json"),
                                      Path("data/prefix-pilot/features.npz"),ref)
    raw,adjusted={},{}
    folds=[]
    for valid in groups.values():
        train=filtered_training(gold,valid,linked)
        a=np.stack([feat[s] for s in train])
        b=np.stack([feat[s] for s in valid])
        targets=np.array([[c in gold[s] for c in CODES] for s in train],float)
        floor=max(0.,float(np.quantile(a[targets[:,CODES.index(FLOOD)]==0]@projection,.95,method="higher")))
        scores=probabilities(fit(a,targets,.01,250),b)
        changed=scores.copy()
        changed[b@projection>=0,CODES.index(FLOOD)]=1.
        raw.update(decisions(valid,changed,.5))
        changed=scores.copy()
        changed[b@projection>floor,CODES.index(FLOOD)]=1.
        adjusted.update(decisions(valid,changed,.5))
        folds.append({"training_threshold":floor,"external_same_test":binary(ytest,ext>floor)})
    return {"scope":"post-pilot, same exposed contest cohort, zero contest inundation positives",
            "official_unadjusted":evaluate(gold,raw),"official_calibrated":evaluate(gold,adjusted),"folds":folds,
            "always_positive_external":binary(ytest,np.ones_like(ytest)),
            "shuffled_1000_external_f1_quantiles":dict(zip(("p025","median","p975"),np.quantile(null,[.025,.5,.975]))),
            "shuffled_at_least_observed":int((null>=base["f1"]).sum()),
            "external_positive_recall_wilson":wilson(base["tp"],base["tp"]+base["fn"]),
            "external_specificity_wilson":wilson(base["tn"],base["tn"]+base["fp"])}


if __name__=="__main__":
    report=run()
    Path("results/external_shift.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print("adjusted flood false positives",report["official_calibrated"]["per_code"][FLOOD]["fp"])
