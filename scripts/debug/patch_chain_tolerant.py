"""Make chain building tolerant: at junctions pick the same-lane successor, and clamp goal."""
from pathlib import Path

g = Path(__file__).resolve().parents[2]/'scripts/carla_scenario_sim.py'
s = g.read_text(encoding='utf8')

helper = '''
def extend_forward(chain, i_ego, need_m):
    """Extend along the lane; at a junction keep the same lane_id instead of failing."""
    while (len(chain)-i_ego-1)*2.0 < need_m:
        nxt = chain[-1].next(2.0)
        if not nxt:
            break
        lane = chain[-1].lane_id
        same = [w for w in nxt if w.lane_id == lane]
        chain.append((same or nxt)[0])
    return chain

'''

anchor = 'def xy_wp(wp):'
assert anchor in s
s = s.replace(anchor, helper.lstrip('\n')+anchor, 1)

old1 = '''        chain = list(reversed(back))+[ego_wp]
        i_ego = len(chain)-1
        while (len(chain)-i_ego-1)*2.0 < cfg['grid']['route_length_m']:
            nxt = chain[-1].next(2.0)
            if len(nxt) != 1:
                raise RuntimeError('ROAD_CHAIN_BROKEN')
            chain.append(nxt[0])'''
new1 = '''        chain = list(reversed(back))+[ego_wp]
        i_ego = len(chain)-1
        chain = extend_forward(chain, i_ego, cfg['grid']['route_length_m'])'''
assert old1 in s
s = s.replace(old1, new1)

old2 = '''                    chain = list(reversed(back))+[ego_wp]
                    i_ego = len(chain)-1
                    while (len(chain)-i_ego-1)*2.0 < cfg['grid']['route_length_m']:
                        nxt = chain[-1].next(2.0)
                        if len(nxt) != 1:
                            raise RuntimeError('ROAD_CHAIN_BROKEN')
                        chain.append(nxt[0])
                    break'''
new2 = '''                    chain = list(reversed(back))+[ego_wp]
                    i_ego = len(chain)-1
                    chain = extend_forward(chain, i_ego, cfg['grid']['route_length_m'])
                    break'''
assert old2 in s
s = s.replace(old2, new2)

# clamp goal / obstacle indices to the available chain
old3 = '''        i_goal = min(len(chain)-1, i_ego+int(round(cfg['goal']['distance_m']/2.0)))'''
new3 = '''        avail_m = (len(chain)-i_ego-1)*2.0
        goal_m = min(cfg['goal']['distance_m'], max(12.0, avail_m-6.0))
        if goal_m < cfg['goal']['distance_m']:
            print('路段只有 %.0f m，目标点缩短到 %.0f m' % (avail_m, goal_m), flush=True)
        i_goal = min(len(chain)-1, i_ego+int(round(goal_m/2.0)))'''
assert old3 in s
s = s.replace(old3, new3)

g.write_text(s, encoding='utf8')
print('chain tolerant + goal clamp')
