# 用 drawio 画这张图的交接说明

给新对话用。读完这份就能安全地继续改图，不用重新摸索。

---

## 一、文件地图

| 文件 | 性质 | 能否被脚本覆盖 |
|---|---|---|
| `TrajSafe-Diffuser_arch-main.drawio` | **你在 drawio 里手工排版的主文件** | **绝对不能** |
| `TrajSafe-Diffuser_arch-main.framed.drawio` | 加了分区框 / 配色 / 线条分级的版本 | 可以（这是脚本产物） |
| `TrajSafe-Diffuser_arch-main.generated.drawio` | `_gen_main_arch.py` 生成的完整参考版 | 可以（脚本只写这个） |
| `TrajSafe-Diffuser_arch-main-bk*.drawio` | 你自己的历史备份 | 不要动 |
| `backup/` | 脚本每次写操作前自动备份（12 个） | 只增不改 |
| `assets/` | 素材 PNG：`n_*.png` 场景图、`t_*.png` token glyph、`a_temb.png` 波形 | 由管线生成 |

**当前状态**：`main` = 36 顶点 / 35 边；`framed` = 40 顶点 / 32 边。

---

## 二、工具脚本（全在 `docs/diagrams/`）

| 脚本 | 用途 | 用法 |
|---|---|---|
| `_render.py` | **按文档顺序**渲染预览图 + 查断边/宽高比 | `python docs/diagrams/_render.py <file.drawio>` → 输出 `<file>.preview.png` |
| `_verify.py` | XML 合法性、页尺寸、顶点/边/图片数、节点重叠 | `python docs/diagrams/_verify.py` |
| `_dump.py` | 打印所有顶点（id/坐标/文字）和边（id/两端/样式） | `python docs/diagrams/_dump.py` |
| `_cellsdiff.py` | **对比两个文件的顶点/边差异** —— 确认"只改了想改的" | `python docs/diagrams/_cellsdiff.py A.drawio B.drawio` |
| `_slack.py` | 量每个容器的内部空白 | — |
| `_imgaudit.py` | 查每个图片 cell 嵌的是哪张素材 | — |
| `_add_frames.py` | 生成带分区框的副本（**只读 main，只写 framed**） | `--apply` |
| `_tint_frames.py` | 给分区框上色 | `--apply` |
| `_tier_edges.py` | 给边分级（粗细/颜色/虚线） | `--apply` |
| `_swapbitmaps.py` | 替换嵌入的图片（可指定 cell） | `python ... _swapbitmaps.py i_occ i_qt` |
| `_del_edge.py` | 删一条边 | `python ... _del_edge.py <edge-id> --apply` |
| `_merge_pair.py` | 把两个相邻框合并（含边重指向） | `--apply` |
| `_rename_box.py` | 改一个框的文字 | `--apply` |
| `_gen_main_arch.py` | 完整生成器（**只写 `.generated.drawio`**） | — |

所有改文件的脚本都遵循同一个模式：**默认 dry run**（只打印要做什么 + 几何检查），加 `--apply` 才写；写前自动备份并 `ET.fromstring()` 验证。

---

## 三、八条铁律

1. **不要用脚本覆盖 `-main.drawio`。** 要改就用 `edit` 工具单点改，或改 `.framed` 副本，或先备份。我曾经用生成脚本覆盖过一次，你手工排的版全丢了，且 untracked 无 git 可恢复。

2. **写之前必须确认 drawio 已关闭。**
   ```powershell
   Get-Process -ErrorAction SilentlyContinue | Where-Object { $_.ProcessName -match 'draw' }
   ```
   不关的话：你按 Ctrl+S 覆盖脚本的改动，或者脚本覆盖你未保存的改动。**双向都会丢。**

3. **绝对不要用 PowerShell `Set-Content` 改含中文的文件。** 它会用系统 ANSI(GBK) 回写，把 UTF-8 中文毁成非法字节，**不可逆**。用 Python：
   ```python
   xml = io.open(p, encoding="utf-8").read()
   io.open(p, "w", encoding="utf-8", newline="").write(xml)
   ```
   要改中文文件就用 `edit` 工具，或写个 `.py` 脚本。

4. **XML 属性里的 `<br>` 必须写成 `&lt;br&gt;`**，否则文件非法（`ET.fromstring` 会报 not well-formed）。

5. **z-order = 文档顺序。** 分区框要画在底层，就必须排在单元格列表最前面（紧跟 `<mxCell id="1" parent="0" />`）。排在后面就会盖住地图。

6. **drawio 每次保存会把属性重排成字母序**，`mxGeometry` 写成 `height width x y`（不是 `x y width height`）。所以**按属性名解析，不要按位置匹配**。

7. **嵌入图片存在 style 里**，格式 `image=data:image/png,<base64>`，**没有 `;base64` 标记** —— style 字符串以 `;` 分隔，加了会截断，图片静默消失。cell 的 `value` 必须是空字符串。

8. **改完必须自证。** 跑 `_verify.py` + `_cellsdiff.py`（拿备份比），确认顶点数、坐标、其余边全都没变。

---

## 四、改图流程模板

```
1. 让用户关掉 drawio，确认无进程
2. 备份 → backup/<name>.<YYYYmmdd-HHMMSS>.drawio
3. 用 Python 定点改：
     - 正则按 id 定位 <mxCell id="X" ...>...</mxCell>
     - 只改目标属性
     - ET.fromstring(xml) 验证
     - 写回
4. python docs/diagrams/_verify.py
   python docs/diagrams/_cellsdiff.py <备份> <当前>
5. python docs/diagrams/_render.py <file>  →  出预览，肉眼确认
6. 告诉用户可以重新打开 drawio
```

---

## 五、查数据流的依据（改图内容时用）

图上的模块和连线必须对得上代码，别凭印象。关键位置：

| 问题 | 去哪查 |
|---|---|
| 模块前向顺序 | `src/models/trajsafe/planner.py` 的 `forward_controls`（约 460-540 行） |
| 每个模块吃哪些输入 | 同名类的 `forward()` 签名 |
| AdaLN / 时步注入 | `grep -n 'AdaLN\|h_t' src/models/trajsafe/` |
| 损失与权重 | `train.py` 的 `DEFAULT_LOSS_WEIGHTS`（约 84 行）+ `LOSS_WEIGHT_KEYS` |
| 几何流水线（凸区域/走廊） | `src/geometry/convex_region.py`、`safety_corridor.py` |

**已确认的两个易错点：**

- **`S,G` 从来没被"编码"。** 它只进两个零参数算子：`hard_control_endpoints(q_t, cond)`（`planner.py:462`）和 `BoundaryDecoder(x̃, cond)`。两个编码器 `encode_trajectory(p_t, c_g, h_t)` / `scene_tokens(occ)` 签名里都没有 `cond`。
- **凸区域构造不含 `h_t`。** `build_from_ellipse(center, shape4)` 和 `build_safety_corridor(...)` 都是纯确定性几何。`h_t` 注入在上游的 `EllipseShapeHead(h_ell, h_t)`（预测 `(a,b,θ)` 那一步）。

---

## 六、踩过的坑（都已修，但别再犯）

| 坑 | 症状 | 修法 |
|---|---|---|
| `_render.py` 先贴所有图、再画所有框 | 带填充的分区框盖住地图（drawio 里却正常） | 改成一条按文档顺序的统一绘制流水线 |
| `_render.py` 把 `<mxPoint as="offset"/>` 当路径点 | `float(None)` 崩溃 | 只读 `<Array as="points">` 里的点，且过滤无 x/y 的 |
| `_render.py` 硬编码线宽和颜色判断 | 看不出粗细差别；`#6B7A99` 判断条件 `"2 3" in st` 是旧样式残留 | 改成按 `strokeWidth` / `strokeColor` 实际取值 |
| `Copy-Item` 复制 `.bkp` | 把源文件的 `Hidden` 属性一起带过去，沙箱拒绝写入 | `attrib -h <file>` |
| 边有 waypoint 但每个只有 `x` 没有 `y` | 我的渲染器崩 | drawio 允许（约束竖向走线），渲染器跳过即可 |

---

## 七、当前遗留

- `framed` 里有 **1 条断边**（`source` / `target` 都是空）：`e2JvZp1_XTm2cSDIqKsH-1`，是删线残骸，可用 `_del_edge.py` 清掉。
- 图例只说明了框（实线框/虚线框/点线框），**没说明线条分级**。建议补：
  `粗黑线 = 主干前向　青色细线 = 条件注入　灰色虚线 = 推理期回环`
- 分区框配色（`_tint_frames.py` 里可调）：
  输入 `#F2F5F8` / ① `#E9F4F8` / ② `#EEEFFB` / ③ `#EBF6EF`
- 线条分级（`_tier_edges.py` 里可调）：主干 `#1F2937` 2.6pt / 条件 `#12738C` 1.4pt / 回环 `#6B7A99` 1.8pt 虚线

---

## 八、和 drawio 配合的注意事项

- 脚本改了文件后，drawio 若已打开会提示"文件已被外部修改"，**选重新加载**，别选覆盖。
- drawio 自己的 `.bkp` 是**保存前**的内容，比被覆盖的版本还早一格，不是可靠备份。
- drawio 桌面版的 Local Storage / Session Storage **不缓存文档内容**（只有 `lastSaveDir` 之类的路径字符串），所以**别指望从那里恢复**。真正的保险是 `backup/` 里的时间戳备份。
- 撤销：drawio 的 Ctrl+Z 只对本次会话有效；文件被外部覆盖后撤销历史就没了。
