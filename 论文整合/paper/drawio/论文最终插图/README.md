# 当前论文使用的 draw.io 插图

本目录集中存放当前论文的五张方法插图，今后编辑请优先打开这里的中文命名文件，不要使用上一级目录的“优化版”或“原排版修订版”。此次仅整理文件位置，没有修改插图内容。

| 编号 | 可编辑文件 | 论文实际引用的渲染图（相对 paper 目录） |
|---|---|---|
| 图 3-1 | [总体框架](TrajSafe-Diffuser_图3-1_总体框架-论文插图版.drawio) | `figures/methods/fig3-1.svg` |
| 图 3-2 | [控制主干与 TrajBlock](TrajSafe-Diffuser_图3-2_控制主干与TrajBlock-论文插图版.drawio) | `figures/methods/publication-v2/fig3-2.svg` |
| 图 3-3 | [骨架匹配与安全几何](TrajSafe-Diffuser_图3-3_骨架匹配与安全几何-论文插图版.drawio) | `figures/methods/publication-v2/fig3-3.svg` |
| 图 3-4 | [融合与最终去噪](TrajSafe-Diffuser_图3-4_融合与最终去噪-论文插图版.drawio) | `figures/methods/publication-v2/fig3-4.svg` |
| 图 3-5 | [安全修正与历史反馈](TrajSafe-Diffuser_图3-5_安全修正与历史反馈-论文插图版.drawio) | `figures/methods/publication-v2/fig3-5.svg` |

图 3-2 至图 3-5 从上一级目录移动到这里，内容 SHA-256 校验一致。图 3-1 从实际出版副本 `paper/figures/methods/drawio/fig3-1.drawio` 原样复制，保留已应用的控制点尺寸与时间标记修订；手工原图、framed 版及旧备份均未移动或覆盖。

`paper/figures/.../drawio/` 下的英文编号文件用于导出，不是另一个设计版本。修改本目录后需同步导出副本及 SVG、PDF、PNG；移动文件本身不会自动修改已导出的图像。四张细节图的生成器 `tools/build_publication_method_diagrams.py` 已改为写入本目录，写前备份仍放在上一级 `backup/`。

此前讨论的图 3-3 三子图展开方案尚未实施，本目录保存的仍是当前论文实际使用的版本。实验数据图及轨迹可视化不是 draw.io 源图，不在本目录中。
