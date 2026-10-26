"""Independently reconstruct accepted endpoints, handovers and internal stops."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def rows(path):
    with path.open(encoding='utf8') as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('directory', type=Path)
    directory = parser.parse_args().directory
    ego = rows(directory/'ego_state.csv')
    by_frame = {r['frame']: r for r in ego}
    requests = {r['frame']:r for r in rows(directory/'planning_requests.csv')}
    candidates = {r['source_frame']:r for r in rows(directory/'plan_candidates.csv')}
    decisions = rows(directory/'trajectory_decisions.csv')
    start = float(ego[0]['sim_time'])
    accepted = []
    cached_paths = {}
    active = None
    for decision in decisions:
        if decision['decision'] != 'ACCEPTED':
            continue
        row = by_frame[decision['frame']]
        request = requests[decision['source_frame']]
        key = decision['source_frame']
        data = np.load(directory/'candidates'/f'{key}.npz')
        points, s = data['curve_world'], data['arc_length_m']
        cached_paths[key] = (points,s)
        old_remaining = None
        if active is not None:
            xy,ss = cached_paths[active]
            index = int(np.argmin(np.linalg.norm(xy-[float(row['x']),float(row['y'])],axis=1)))
            old_remaining = float(ss[-1]-ss[index])
        validation = json.loads(candidates[key].get('dynamic_validation') or '{}')
        accepted.append(dict(source_frame=int(key), sim_seconds=float(row['sim_time'])-start,
            request_seconds=float(request['sim_time'])-start, behavior=decision['behavior'],
            measured_speed_mps=float(row['speed']), remaining_old_path_m=old_remaining,
            endpoint_world_xy_m=[float(request['goal_x']),float(request['goal_y'])],
            endpoint_route_progress_m=request.get('goal_route_progress_m'),
            requested_horizon_m=request.get('requested_horizon_m'), final_goal=request.get('final_goal'),
            request_reason=request['replan_reason'], dynamic_validation=validation))
        active = key
    groups = []
    for row in ego:
        # Exclude initialisation, legitimate waiting and the final destination stop.
        if row.get('active_source_frame') and row.get('active_final_goal') != 'True' and row['behavior'] not in ('PREPARE','WAIT_GAP') and float(row['speed']) < .3:
            if not groups or float(row['sim_time'])-float(groups[-1][-1]['sim_time']) > .075:
                groups.append([])
            groups[-1].append(row)
    stops = [dict(start_s=float(g[0]['sim_time'])-start,end_s=float(g[-1]['sim_time'])-start,
                  remaining_path_m=g[0].get('remaining_path_m'),behavior=g[0]['behavior'])
             for g in groups if float(g[-1]['sim_time'])-float(g[0]['sim_time']) > .3]
    launches = [r for r in stops if accepted and
                abs(r['start_s']-accepted[0]['sim_seconds']) < .075 and
                accepted[0]['measured_speed_mps'] < .3]
    stops = [r for r in stops if r not in launches]
    same_target_cancellations = 0
    for d in decisions:
        if d['decision'] == 'POLICY_CHANGED':
            target = 'left' if by_frame[d['frame']]['behavior'] in ('PREPARE','WAIT_GAP','CHANGE','PASS') else 'original'
            same_target_cancellations += requests[d['source_frame']]['target'] == target
    report = dict(accepted_endpoints=accepted, internal_low_speed_intervals=stops,
                  initial_launch_low_speed_intervals=launches,
                  same_target_goal_version_cancellations=same_target_cancellations,
                  minimum_old_path_remaining_at_handover_m=min((a['remaining_old_path_m'] for a in accepted if a['remaining_old_path_m'] is not None), default=None),
                  note='Dynamic validation is prediction under constant NPC velocity; it is not a guarantee for unknown future manoeuvres.')
    (directory/'handover_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
