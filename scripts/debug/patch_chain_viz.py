"""修复：① 路线向后延伸（车后也要有路）② 保存骨架候选/选中拓扑/椭圆，便于画图。"""
from pathlib import Path

root = Path(__file__).resolve().parents[2]
g = root/'scripts/carla_scenario_sim.py'
s = g.read_text(encoding='utf8')

# ---- 1. 链条向后延伸
old_chain = '''        chain = [ego_wp]
        while (len(chain)-1)*2.0 < cfg['grid']['route_length_m']:
            nxt = chain[-1].next(2.0)
            if len(nxt) != 1:
                raise RuntimeError('ROAD_CHAIN_BROKEN')
            chain.append(nxt[0])'''
new_chain = '''        back_m = float(cfg['grid'].get('route_back_m', 40.0))
        back = []
        probe = ego_wp
        while len(back)*2.0 < back_m:
            prv = probe.previous(2.0)
            if len(prv) != 1:
                break
            probe = prv[0]
            back.append(probe)
        chain = list(reversed(back))+[ego_wp]
        i_ego = len(chain)-1
        while (len(chain)-i_ego-1)*2.0 < cfg['grid']['route_length_m']:
            nxt = chain[-1].next(2.0)
            if len(nxt) != 1:
                raise RuntimeError('ROAD_CHAIN_BROKEN')
            chain.append(nxt[0])'''
assert old_chain in s
s = s.replace(old_chain, new_chain)

# ---- 2. 目标点索引改为相对 ego
old_goal = "        i_goal = min(len(chain)-1, int(round(cfg['goal']['distance_m']/2.0)))"
new_goal = "        i_goal = min(len(chain)-1, i_ego+int(round(cfg['goal']['distance_m']/2.0)))"
assert old_goal in s
s = s.replace(old_goal, new_goal)

# ---- 3. 障碍物索引相对 ego
old_obs = "            j = min(len(chain)-1, int(round(spec['distance_m']/2.0)))"
new_obs = "            j = min(len(chain)-1, i_ego+int(round(spec['distance_m']/2.0)))"
assert old_obs in s
s = s.replace(old_obs, new_obs)

# ---- 4. 把骨架/选中拓扑/椭圆存进 plan.npz
old_save = '''        np.savez_compressed(out/'plan.npz', curve_world=path.xy, controls=result['controls'],
                            occupancy=occupancy, road=road_mask, condition=condition,
                            anchor=frame.origin, yaw=frame.yaw, goal=goal_xy)'''
new_save = '''        viz_d = result.get('viz') or {}
        cand = viz_d.get('candidates') or {}
        np.savez_compressed(
            out/'plan.npz', curve_world=path.xy, controls=result['controls'],
            occupancy=occupancy, road=road_mask, condition=condition,
            anchor=frame.origin, yaw=frame.yaw, goal=goal_xy,
            cand_xy=(cand.get('xy') if cand else np.zeros((0, 2))),
            cand_mask=(cand.get('mask') if cand else np.zeros(0, bool)),
            selected=int(viz_d.get('selected', -1)),
            cand_geom=(cand.get('geometry') if cand else np.zeros((0, 2))),
            cand_len=(cand.get('lengths') if cand else np.zeros(0)),
            corridors=np.array([np.asarray(p) for p in (viz_d.get('corridor') or {}).get('polygons', [])],
                               dtype=object),
            ellipse_center=np.asarray(result.get('ellipse_center') or [], dtype=float),
            ellipse_ab=np.asarray(result.get('ellipse_ab') or [], dtype=float))'''
assert old_save in s
s = s.replace(old_save, new_save)
g.write_text(s, encoding='utf8')

# ---- 5. adapter 暴露椭圆
a = root/'src/carla_bridge/planner_adapter.py'
t = a.read_text(encoding='utf8')
if 'ellipse_center' not in t:
    t = t.replace("activation=out['activation_info'], timing=out['timing'])",
                  "activation=out['activation_info'], timing=out['timing'],\n"
                  "                    ellipse_center=out.get('ellipse_center'),\n"
                  "                    ellipse_ab=(None if out.get('ellipse_a') is None else\n"
                  "                                np.stack([out['ellipse_a'], out['ellipse_b']], axis=-1)))")
    a.write_text(t, encoding='utf8')
print('runner/adapter 已修：链条向后延伸 + 存骨架/椭圆')
