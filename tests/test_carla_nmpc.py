"""已随 2026-10 简化删除：NMPC 基线。

闭环代码现只保留「CARLA 取数 → 占据图 → 论文方法出轨迹」三步，
执行控制、行为状态机、安全检查与在线约束全部移除，对应测试随之作废。
入口见 scripts/carla_plan_once.py。

本文件保留仅为让清理身份删不掉时仍可手工删除。
"""
import pytest

pytest.skip("closed-loop execution/safety layer removed", allow_module_level=True)
