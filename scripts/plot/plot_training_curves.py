"""Plot training curves from train.log."""
import argparse, re, ast, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--log", default="logs/train.log",
                help="path to the training log (default logs/train.log)")
ap.add_argument("--out", default="outputs/training_curves.png",
                help="output image path")
args = ap.parse_args()
path = args.log
train={}; val={}
it=re.compile(r"epoch (\d+)/\d+ loss=({.*?}) time")
iv=re.compile(r"val\s+loss=({.*?})")
for line in open(path, encoding="utf-8"):
    m=it.search(line)
    if m:
        ep=int(m.group(1)); d=ast.literal_eval(m.group(2)); train[ep]=d; continue
    m2=iv.search(line)
    if m2:
        # val line appears right after train; associate with last seen epoch
        d=ast.literal_eval(m2.group(1));
        if train:
            ep=max(train); val[ep]=d
eps=sorted(train)
keys=["total","L_diff","L_ellipse","L_param","L_iou","L_ecol_raw","L_anchor","L_col","L_smooth"]
fig,axes=plt.subplots(3,1,figsize=(12,14))
# 1) total
ax=axes[0]
ax.plot(eps,[train[e].get("total") for e in eps],label="train total",marker="o")
ax.plot(eps,[val.get(e,{}).get("total") for e in eps if e in val],label="val total",marker="s")
ax.set_ylabel("total"); ax.legend(); ax.grid(alpha=0.3); ax.set_title(f"Training curves (mixed-scene, {len(eps)} epochs)")
# 2) main components (train)
ax=axes[1]
for k in ["L_diff","L_ellipse","L_col","L_smooth"]:
    ax.plot(eps,[train[e].get(k,0) for e in eps],label=k,marker="o")
ax.set_ylabel("loss"); ax.legend(); ax.grid(alpha=0.3)
# 3) ellipse sub-losses (train)
ax=axes[2]
for k in ["L_param","L_iou","L_ecol_raw","L_anchor"]:
    ax.plot(eps,[train[e].get(k,0) for e in eps],label=k,marker="o")
ax.set_ylabel("loss"); ax.set_xlabel("epoch"); ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
os.makedirs(os.path.dirname(args.out) or ".",exist_ok=True)
out=args.out
fig.savefig(out,dpi=120,bbox_inches="tight")
print("saved",out,"epochs",len(eps),"(log:",path,")")
print("final train",{k:round(train[eps[-1]][k],4) for k in keys if k in train[eps[-1]]})
print("final val",{k:round(val[eps[-1]][k],4) for k in keys if k in val.get(eps[-1],{})})
