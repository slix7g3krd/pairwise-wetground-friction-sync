# pairwise-wetground-friction-sync

## 领域

游戏、图形与多媒体

## 主题

湿滑地面与摩擦同步

## 说明

固定tick累积、物理渲染同源与材质分档。

初始状态下 `gameplay/wetground_friction_sync.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
