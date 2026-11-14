import torch, json
for tag, p in (('A-C48', 'outputs/models/oneshot_k4p_c48/ckpt/best_task.pt'),
               ('S-C48', 'outputs/models/oneshot_k4p_c48_s/ckpt/best_task.pt'),
               ('D-C48', 'outputs/models/oneshot_k4p_c48_d/ckpt/best_task.pt')):
    d = torch.load(p, map_location='cpu', weights_only=False)
    c = d['config']
    print('=' * 66)
    print(tag, 'epoch', d['epoch'])
    print('  loss:', json.dumps(c.get('loss'), ensure_ascii=False))
    print('  ellipse_label:', json.dumps(c.get('ellipse_label'), ensure_ascii=False)[:400])
    print('  model.safety_features:', json.dumps((c.get('model') or {}).get('safety_features'), ensure_ascii=False))
    print('  train.feedback.rollout:', (c.get('train') or {}).get('feedback'))
