# -*- coding: utf-8 -*-
"""Cleaner architecture diagram (fewer crossings)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei","SimHei","DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
fig, ax = plt.subplots(figsize=(12,14))
ax.set_xlim(0,12); ax.set_ylim(0,14); ax.axis("off")
def box(x,y,w,h,t,fc="#eef4fb",ec="#3b6ea5",fs=9):
    ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle="round,pad=0.02",lw=1.5,ec=ec,fc=fc))
    ax.text(x,y,t,ha="center",va="center",fontsize=fs)
def arrow(p1,p2,text=None,c="#3b6ea5",fs=8):
    ax.add_patch(FancyArrowPatch(p1,p2,arrowstyle="-|>",mutation_scale=13,lw=1.3,color=c))
    if text: ax.text((p1[0]+p2[0])/2,(p1[1]+p2[1])/2,text,ha="center",va="center",fontsize=fs,color="#555")

# 输入
box(1.8,13.3,2.4,0.7,"地图 map","#fde9d9","#c96a1a")
box(6.0,13.3,2.4,0.7,"扩散态 z_t","#fde9d9","#c96a1a")
box(10.2,13.3,2.6,0.7,"cond start/goal","#fde9d9","#c96a1a")

# 左：场景
box(1.8,11.6,3.0,0.9,"SceneEncoder (U-Net)","#e8f5e9","#2e7d32")
arrow((1.8,13.3),(1.8,12.05))
box(1.8,9.9,3.0,0.9,"memory 16×16 → scene_tokens[256]\n(地图正弦2D)","#e8f5e9","#2e7d32",fs=8)
box(3.6,9.9,2.4,0.9,"local 256×256\n(地图正弦2D)","#e8f5e9","#2e7d32",fs=8)
arrow((1.8,11.15),(1.8,10.35)); arrow((1.8,11.15),(3.6,10.35))

# 中：轨迹
box(6.0,11.6,3.4,0.9,"TrajectoryEncoder\n(自注意力×2)","#e3f2fd","#1565c0")
arrow((6.0,13.3),(6.0,12.05),"z_t+序号k+时间t")
box(6.0,9.9,3.4,0.7,"F_traj [B,127,C]","#e3f2fd","#1565c0")
arrow((6.0,11.15),(6.0,10.3))

# 位置恢复（并入一个小块，靠中右）
box(8.2,8.2,3.2,0.9,"恢复位置\nΔp=g/N+z_t; pos_t=积分; p_k=pos_t[:,1:]","#fff9c4","#b58900",fs=8)
arrow((6.0,9.55),(8.2,8.65),"z_t→位置","#b58900")

# 局部采样 + 交叉注意力（右）
box(8.2,6.1,3.2,0.9,"LocalSceneSampler(9×9)\n(只取地图特征)","#f3e5f5","#8e24aa",fs=8)
arrow((3.6,9.45),(8.2,6.55),"local","#8e24aa")
arrow((8.2,7.75),(8.2,6.55),"p_k","#b58900")
box(8.2,4.2,3.2,0.9,"PointSceneAttention\n(交叉注意力 q=F_traj, kv=local)","#f3e5f5","#8e24aa",fs=8)
arrow((8.2,5.65),(8.2,4.65),"local_scene")
arrow((6.0,9.55),(8.2,4.65),"F_traj","#1565c0")

# 融合
box(6.0,2.7,3.4,0.9,"SafetyFusion\n[F_traj;A_t]→S_t","#e8eaf6","#3949ab")
arrow((8.2,3.75),(7.6,3.1),"A_t","#8e24aa")
arrow((6.0,9.55),(6.0,3.1),"F_traj","#1565c0")

# 条件记忆 C（右上）
box(10.2,8.2,3.0,0.9,"条件记忆 C\n[start;goal;256 scene]\n(start/goal 正弦2D)","#ede7f6","#4527a0",fs=8)
arrow((10.2,13.3),(10.2,8.65),"","#4527a0")
arrow((1.8,9.45),(8.7,8.2),"scene_tokens","#4527a0")

# 分支：椭圆
box(2.6,1.0,3.0,0.9,"EllipseHead (MLP)\n→ center/r1/r2/θ","#fce4ec","#c2185b",fs=8)
arrow((6.0,2.25),(2.6,1.45),"S_t","#c2185b")

# 分支：解码器
box(8.2,1.0,3.0,0.9,"TrajectoryDecoder\n(自注意力+交叉注意力×4)\nQ=S_t,K=V=C","#e3f2fd","#1565c0",fs=8)
arrow((6.0,2.25),(8.2,1.45),"S_t(query)","#1565c0")
arrow((10.2,7.75),(8.2,1.45),"C(memory)","#4527a0")
box(8.2,-0.5,3.0,0.6,"ResidualHead→z0→积分→轨迹","#e3f2fd","#1565c0",fs=8)
arrow((8.2,0.55),(8.2,0.0))

fig.suptitle("Neural-IRISDiffuser 整体框架（自注意力/交叉注意力）", fontsize=13)
import os
os.makedirs("outputs",exist_ok=True)
plt.tight_layout(); plt.savefig("outputs/architecture.png",dpi=130,bbox_inches="tight")
print("saved outputs/architecture.png")
