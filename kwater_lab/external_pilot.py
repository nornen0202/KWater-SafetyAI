"""Run the preregistered external-image comparisons on CPU; no private images leave the machine."""

import argparse
import ast
from contextlib import contextmanager
from collections import Counter, defaultdict
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image
from tokenizers import Tokenizer

from .clip_probe import assets, DESCRIPTIONS, ImageEncoder, session, unit_rows, REVISION
from .metrics import CODES, evaluate
from .prefix_pilot import filtered_training
from .train_head import decisions, fit, probabilities, save_atomic
from .weight_pilot import pilot_data


ROOT = Path("data/external")
FLOOD = "drowning_inundation"
PROMPTS = ["an aerial photograph of flooded roads and buildings after a flood",
           "an aerial photograph of dry passable roads and buildings"]


def picture(archive, name):
    with Image.open(BytesIO(archive.read(name))) as im:
        return im.convert("RGB")


def dhash(im):
    grid = np.array(im.convert("L").resize((9,8), Image.Resampling.LANCZOS))
    return int.from_bytes(np.packbits(grid[:,1:] > grid[:,:-1]).tobytes(), "big")


def unit(x):
    return x / max(float(np.linalg.norm(x)), 1e-12)


def axis(x, y):
    return unit(x[y].mean(0)) - unit(x[~y].mean(0))


def text_vectors(model,prompts):
    tokenizer = Tokenizer.from_file(str(model/"tokenizer.json"))
    language = session(model/"text_model_quantized.onnx")
    vectors=[]
    for prompt in prompts:
        tokens = np.array([tokenizer.encode(prompt).ids],dtype=np.int64)
        vectors.append(language.run(["text_embeds"],{"input_ids":tokens})[0][0])
    return unit_rows(np.stack(vectors))


def binary(y, predicted):
    pred = np.asarray(predicted,dtype=bool)
    truth = np.asarray(y,dtype=bool)
    tp, fp = int((pred & truth).sum()), int((pred & ~truth).sum())
    fn, tn = int((~pred & truth).sum()), int((~pred & ~truth).sum())
    def rat(a,b):
        return float(a/b) if b else None
    return {"tp":tp,"tn":tn,"fp":fp,"fn":fn,"f1":rat(2*tp,2*tp+fp+fn),
            "recall":rat(tp,tp+fn),"specificity":rat(tn,tn+fp),"accuracy":rat(tp+tn,len(y))}


def hashed(key):
    return hashlib.sha256(key.encode()).digest()


@contextmanager
def readable_archive(path):
    """Allow the downloader to rename a fully written file while this reader holds it."""
    if os.name != "nt":
        with path.open("rb") as f:
            yield f
        return
    import ctypes
    import msvcrt
    from ctypes import wintypes
    open_shared = ctypes.WinDLL("kernel32", use_last_error=True).CreateFileW
    open_shared.argtypes = [wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,wintypes.LPVOID,
                            wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE]
    open_shared.restype = wintypes.HANDLE
    handle = open_shared(str(path.resolve()),0x80000000,7,None,3,0x80,None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    with os.fdopen(msvcrt.open_osfhandle(handle,os.O_RDONLY | os.O_BINARY),"rb") as f:
        yield f


def flood_rows(z, split, cap, encoder, excluded):
    base = f"FloodNet-Supervised_v1.0/{split}/{split}-"
    labels = {Path(n).stem.removesuffix("_lab"):n for n in z.namelist() if n.startswith(base+"label-img/") and n.endswith(".png")}
    images = {Path(n).stem:n for n in z.namelist() if n.startswith(base+"org-img/") and n.endswith(".jpg")}
    if set(images) != set(labels):
        raise ValueError("Unpaired FloodNet images")
    selected = []
    counts = Counter()
    rejected = Counter()
    ids = sorted(images, key=hashed)
    for sid in ids:
        with Image.open(BytesIO(z.read(labels[sid]))) as mask:
            if mask.mode not in ("L","P"):
                raise ValueError("Expected indexed mask")
            grid = np.asarray(mask.resize((128,128), Image.Resampling.NEAREST))
        if grid.max()>9:
            raise ValueError("Unknown mask class")
        wet = np.isin(grid,[1,3]).mean()
        neg = np.isin(grid,[2,4,5,8]).mean()
        if wet>=0.02:
            label = True
        elif wet==0 and neg>=0.02:
            label = False
        else:
            rejected["ambiguous"] += 1
            continue
        if counts[label] >= cap:
            continue
        im = picture(z,images[sid])
        finger = dhash(im)
        if any((finger ^ other).bit_count()<=4 for other in excluded):
            rejected["near_duplicate_train"] += 1
            continue
        selected.append((encoder.encode(im,"full_frame"),label,finger,sid))
        counts[label] += 1
        if min(counts[True],counts[False])>=cap:
            break
    if not counts[True] or not counts[False]:
        raise ValueError("No positives or negatives")
    return selected, {"available_pairs":len(images),"positive":counts[True],"negative":counts[False],
                      "rejected_during_selection":dict(rejected)}


def flood(encoder, model, cache):
    zpath = ROOT/"floodnet/FloodNet.zip"
    receipt_path = zpath.with_suffix(".zip.json")
    staged = zpath if zpath.exists() else zpath.with_suffix(".zip.partial")
    if staged.stat().st_size != 13057835433:
        raise ValueError("FloodNet transfer has not reached its published byte count")
    with readable_archive(staged) as stream, ZipFile(stream) as z:
        directory = hashlib.sha256("\n".join(f"{i.filename}/{i.file_size}/{i.CRC}" for i in z.infolist()).encode()).hexdigest()
    sig = hashlib.sha256((REVISION+directory+"flood-grid128-v1").encode()).hexdigest()
    if cache.exists():
        with np.load(cache,allow_pickle=False) as a:
            if a["sig"].item() != sig:
                raise ValueError("External feature cache fingerprint changed")
            features, truth, split = a["features"],a["truth"],a["split"]
            audit = json.loads(a["audit"].item())
    else:
        with readable_archive(zpath if zpath.exists() else staged) as stream, ZipFile(stream) as z:
            print("Selecting FloodNet training masks",flush=True)
            train, tr = flood_rows(z,"train",96,encoder,[])
            print("Selecting FloodNet test masks",flush=True)
            test, te = flood_rows(z,"test",64,encoder,[r[2] for r in train])
        features = np.stack([r[0] for r in train+test])
        truth = np.array([r[1] for r in train+test], dtype=bool)
        split = np.array([0]*len(train)+[1]*len(test))
        audit = {"train":tr,"test":te,"selection_sha256":hashlib.sha256(("\n".join(r[3] for r in train+test)).encode()).hexdigest()}
        save_atomic(cache, lambda f:np.savez_compressed(f,sig=sig,features=features,truth=truth,split=split,audit=json.dumps(audit)))
    x,y = features[split==0],truth[split==0]
    test,want = features[split==1],truth[split==1]
    pos, neg = text_vectors(model,PROMPTS)
    learned = axis(x,y)
    stats = {"always_negative":binary(want,np.zeros(len(test))),"fixed_text":binary(want,test@(pos-neg)>=0),
             "floodnet_centroid":binary(want,test@learned>=0),
             "shuffled_train_labels":binary(want,test@axis(x,np.random.default_rng(42).permutation(y))>=0)}
    digest = json.loads(receipt_path.read_text())["sha256"] if receipt_path.exists() else None
    return learned, {**audit,"metrics":stats,"threshold":0,"source_sha256":digest,"directory_sha256":directory}


def kalis(encoder):
    classes = defaultdict(list)
    all_boxes = Counter()
    hashes = set()
    invalid = empty = duplicates = 0
    with ZipFile(ROOT/"kalis/sample.zip") as z:
        if z.testzip():
            raise ValueError("KALIS CRC failure")
        names = ast.literal_eval(next(s.split(":",1)[1] for s in z.read("data.yaml").decode("utf-8-sig").splitlines() if s.startswith("names:")))
        files = {Path(s).stem:s for s in z.namelist() if s.lower().endswith((".jpg",".png"))}
        labels = {Path(s).stem:s for s in z.namelist() if "/labels/" in s and s.endswith(".txt")}
        if set(files)!=set(labels):
            raise ValueError("Unpaired KALIS data")
        for sid,file in sorted(files.items()):
            im = picture(z,file)
            h = hashlib.sha256(im.tobytes()).digest()
            duplicates += h in hashes
            hashes.add(h)
            lines=z.read(labels[sid]).decode("utf-8-sig").splitlines()
            empty+=not any(s.strip() for s in lines)
            for line in filter(str.strip,lines):
                fields = line.split()
                if len(fields)!=5:
                    invalid+=1
                    continue
                c, (x,y,w,h) = int(fields[0]),map(float,fields[1:])
                edges=np.array([x-w/2,y-h/2,x+w/2,y+h/2])
                if not 0<=c<len(names) or not np.isfinite(edges).all() or w<=0 or h<=0 or edges.min() < -1e-5 or edges.max()>1.00001:
                    invalid+=1
                    continue
                box=np.rint(np.clip(edges,0,1)*np.array([im.width,im.height]*2)).astype(int)
                if box[2]<=box[0] or box[3]<=box[1]:
                    invalid+=1
                    continue
                classes[names[c]].append(encoder.encode(im.crop(tuple(box)),"full_frame"))
                all_boxes[names[c]]+=1
    centers = np.stack([unit(np.mean(classes[k],axis=0)) for k in sorted(classes)])
    return centers, {"pairs":len(files),"label_names":names,"valid_boxes_by_class":dict(sorted(all_boxes.items())),
                     "invalid_boxes":invalid,"empty_labels":empty,"exact_image_duplicates":duplicates}


def official(k_centers, flood_axis):
    reference=json.loads(Path("results/prefix_pilot_fpr20.json").read_text())
    groups,feat,gold,linked=pilot_data(Path("results/train_audit_full.json"),Path("data/prefix-pilot/exposure.json"),
                                      Path("data/prefix-pilot/features.npz"),reference)
    preds={a:{} for a in ("official_only","kalis","floodnet","kalis_and_floodnet")}
    for valid in groups.values():
        tr=filtered_training(gold,valid,linked)
        train=np.stack([feat[s] for s in tr])
        test=np.stack([feat[s] for s in valid])
        targets=np.array([[c in gold[s] for c in CODES] for s in tr],float)
        for k,(plain,plus) in enumerate([("official_only","floodnet"),("kalis","kalis_and_floodnet")]):
            a,b=train,test
            if k:
                a,b = np.column_stack([a,a@k_centers.T]),np.column_stack([b,b@k_centers.T])
            scores=probabilities(fit(a,targets,0.01,250),b)
            preds[plain].update(decisions(valid,scores,.5))
            changed=scores.copy()
            changed[test@flood_axis>=0,CODES.index(FLOOD)]=1.
            preds[plus].update(decisions(valid,changed,.5))
    return {k:evaluate(gold,p,{s:g for g,ids in groups.items() for s in ids}) for k,p in preds.items()}


def construction(encoder,model):
    with ZipFile(ROOT/"aihub/71407/sample.zip") as z:
        if z.testzip():
            raise ValueError("71407 sample CRC failure")
        images={Path(s).stem:s for s in z.namelist() if s.endswith(".jpg")}
        labels={Path(s).stem:s for s in z.namelist() if s.endswith(".json")}
        if set(images)!=set(labels):
            raise ValueError("Unpaired construction data")
        x,y,sites,clips,risks=[],[],[],[],[]
        cells=Counter()
        for sid,n in sorted(images.items()):
            label=json.loads(z.read(labels[sid]))
            if label["Source_Data_Info."]["Source_Data"]!=sid:
                raise ValueError("Source annotation ID mismatch")
            parts=n.split("/")
            if "5대사고유형" not in parts:
                continue
            _,risk,state=parts[2:5]
            if state not in ("정상","비정상"):
                raise ValueError("Unknown construction state")
            y.append(state=="비정상")
            sites.append(str(label["Raw_Data_Info."]["Location_ID"]))
            clips.append(str(label["Raw_Data_Info."]["Raw_data_ID"]))
            risks.append(risk)
            cells[risk+"_"+state]+=1
            x.append(encoder.encode(picture(z,n),"full_frame"))
    x,y,sites,clips=np.stack(x),np.array(y,bool),np.asarray(sites),np.asarray(clips)
    pred,random=np.zeros_like(y),np.zeros_like(y)
    for site in sorted(set(sites)):
        test=sites==site
        train=(~test)&~np.isin(clips,clips[test])
        if len(set(y[train]))!=2:
            raise ValueError("No training support in construction split")
        pred[test]=x[test]@axis(x[train],y[train])>=0
        random[test]=x[test]@axis(x[train],np.random.default_rng(42).permutation(y[train]))>=0
    # Each normal row is negative only for its annotated hazard, not for every class.
    mapping={"추락":"fall_height","전도":"slip_trip","협착":"caught_in_machine",
             "낙하":"struck_by_lifting","화재":"fire_explosion"}
    codes=[*CODES,"no_visible_hazard"]
    prompts=[f"A photo of an industrial worksite with {DESCRIPTIONS[k]}." for k in codes]
    texts=text_vectors(model,prompts)
    scores=x@texts.T
    ranking=np.argsort(-scores[:,:len(CODES)],axis=1,kind="stable")
    selected=ranking[:,:3]
    target=np.array([CODES.index(mapping[k]) for k in risks])
    has_target=(selected==target[:,None]).any(axis=1) & (scores[np.arange(len(y)),target]>scores[:,-1])
    return {"pairs":len(images),"state_labeled":len(y),"source_videos":len(set(clips)),"sites_held_out":len(set(sites)),
            "by_type_and_state":dict(cells),"centroid":binary(y,pred),"shuffled":binary(y,random),
            "exploratory_frozen_11_code_probe":{"rule":"top three hazard scores above no-hazard prompt; evaluate only the annotated code per image",
                                               "pooled":binary(y,has_target),
                                               "per_code":{mapping[k]:binary(y[np.array(risks)==k],has_target[np.array(risks)==k]) for k in sorted(set(risks))}},
            "by_type":{k:binary(y[np.array(risks)==k],pred[np.array(risks)==k]) for k in sorted(set(risks))}}


def busan():
    with ZipFile(ROOT/"aihub/71793/sample.zip") as z:
        if z.testzip():
            raise ValueError("Busan CRC failure")
        images={Path(s).stem:s for s in z.namelist() if s.endswith(".jpg")}
        labels={Path(s).stem:s for s in z.namelist() if s.endswith(".json")}
        if set(images)!=set(labels):
            raise ValueError("Unpaired Busan data")
        types,levels,region_levels=Counter(),Counter(),Counter()
        groups=set()
        file_mismatch=0
        for sid,n in sorted(labels.items()):
            obj=json.loads(z.read(n))
            family=n.split("/")[2]
            types[family]+=1
            if "지역 라벨링" not in family:
                continue
            groups.add(Path(n).parent.as_posix())
            file_mismatch+=not any(Path(i["file_name"]).stem==sid for i in obj.get("images",[]))
            meta=next((v for v in obj.values() if isinstance(v,dict) and "danger_level" in v),{})
            level=str(meta.get("danger_level","missing"))
            levels[level]+=1
            region_levels[n.split("/")[3]+" / "+level]+=1
    return {"pairs":len(images),"families":dict(types),"cctv_folder_groups":len(groups),
            "cctv_image_field_mismatches":file_mismatch,"cctv_provider_danger_level":dict(levels),
            "cctv_region_level_cross_tab":dict(region_levels)}


def highwork():
    folder=ROOT/"aihub/507/packages"
    with ZipFile(next(folder.glob("68035-*.zip"))) as src, ZipFile(next(folder.glob("68023-*.zip"))) as lab:
        images={Path(n).stem:n for n in src.namelist() if n.lower().endswith(".jpg")}
        labels={Path(n).stem:n for n in lab.namelist() if n.lower().endswith(".json")}
        matched=sorted(images.keys() & labels.keys(),key=hashed)
        classes,states=Counter(),Counter()
        clips=set()
        mismatches=invalid=0
        for sid in matched:
            obj=json.loads(lab.read(labels[sid]))
            raw=obj["Raw Data Info."]
            clips.add(str(raw["raw_data_ID"]))
            states[raw["situation_ID"].split("-")[0]]+=1
            w,h=raw["resolution"]
            mismatches+=obj["Source Data Info."]["source_data_ID"]!=sid
            for a in obj["Learning Data Info."]["annotation"]:
                if "box" not in a:
                    continue
                classes[a["class_id"]]+=1
                x,y,bw,bh=a["box"]
                invalid+=not (np.isfinite([x,y,bw,bh]).all() and bw>0 and bh>0 and
                              x>=-.01 and y>=-.01 and x+bw<=w+.01 and y+bh<=h+.01)
        checked_size=0
        for sid in matched[:32]:
            im=picture(src,images[sid])
            truth=json.loads(lab.read(labels[sid]))["Raw Data Info."]["resolution"]
            checked_size+=list(im.size)==truth
        return {"scope":"original validation rope-work partition","files":{"raw":"68035","labels":"68023"},
                "images":len(images),"labels":len(labels),"matched":len(matched),"unique_raw_videos":len(clips),
                "source_id_mismatches":mismatches,"provider_states":dict(states),"box_classes":dict(classes),
                "invalid_xywh_boxes":int(invalid),"decoded_images":min(32,len(matched)),"decoded_size_matches":checked_size}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",type=Path,default=Path("G:/CodexData/kwater-rd-20261125/clip-vit-base-patch32"))
    parser.add_argument("--part",choices=("construction","flood","busan","kalis","highwork"),required=True)
    opt=parser.parse_args()
    if opt.part=="busan":
        report=busan()
    elif opt.part=="highwork":
        report=highwork()
    else:
        assets(opt.model)
        encoder=ImageEncoder(opt.model,verify_vision=True)
        if opt.part=="construction":
            report=construction(encoder,opt.model)
        elif opt.part=="kalis":
            _,report=kalis(encoder)
        else:
            flood_axis,external=flood(encoder,opt.model,ROOT/"floodnet/pilot-features.npz")
            centers,k=kalis(encoder)
            report={"external":external,"kalis":k,"official_pilot":official(centers,flood_axis)}
    target=Path("results")/("external_"+opt.part+".json")
    save_atomic(target,lambda f:f.write((json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n").encode()))
    print(target,flush=True)


if __name__=="__main__":
    main()
