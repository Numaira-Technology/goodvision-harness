# 阈值标定协议(SPEC §5/§8 展开)

## 为什么 0.5 对连续分数的模型不公平

片级判定是“帧分数取最大,大于 gate 就算 yes”。

qwen/gemini 每帧只吐 0 或 1,gate 0.5 就等于“任一帧为 true”,没有别的空间可调,调它没意义。luna 的 `noul` 和 CLIP 的 margin 不一样,它们每帧给的是一个连续分数,0.5 是从二值逻辑借来的边界,不一定是这个模型真正该用的工作点,得靠数据定。

“取最大”这一步对正例和负例的影响相反:帧抽得多,正例更容易被抓到(召回高),负例也更容易误触发(误报高)。某个事件如果每帧误报偏高(luna 的 fall 就是),在 0.5 下误报会被放大,把这个事件的阈值抬高就能压下去。

## 做法:留出集选阈 → 冻结 → 测试集评估

1. 留出集:一组不在任何测试集里的片子(`validation_split.json`,`disjoint_assert` 全 0),只在这上面选阈。
2. 按事件扫阈:τ 从 0 到 1 每 0.01 试一遍,算验证集准确率,取并列最优那段的中点。`tune_threshold.py` 的 `pick_tau` 就是干这个。
3. 把选出的 per-class τ 冻结。
4. 用冻结的 τ 在测试集上算分,和 0.5 基线并排报。
5. 主表永远是 0.5,校准结果只作脚注或变体,不替换主表,也不进 Drive 冻结集的主结论。

最关键的一条:阈值只能在留出集上选,不能在测试集上选。在测试集上挑一个能把测试分数拉到最高的阈值,等于把答案偷看进了模型配置,报出来的分虚高、换一批数据就垮。留出集的意义就是“选阈用的数据”和“报分用的数据”彻底分开。

## 实例(luna,v1)

留出集 24 片(fight/smoking/fall/weapon 各 6,12 正 12 负,1fps),与 50/100 测试集零重叠。选出的阈值:

| 事件 | τ | 相对 0.5 | 作用 |
|---|---|---|---|
| fight | 0.29 | 调低 | 多抓打斗 |
| smoking | 0.53 | 基本不变 | — |
| fall | 0.58 | 调高 | 压掉把日常动作当跌倒的误报 |
| weapon | 0.46 | 基本不变 | — |

测试集(冻结 τ vs 0.5):50 片 84% → 94%,100 题 85% → 92%。提升几乎全来自 fall:0.5 下它误报多(50 片 4 个、100 题 7 个),抬到 0.58 后这些误报消失。

## 命令

```sh
# 1) 留出集打分(已有 results/luna_val24.json 且与冻结 split 一致就跳过)
python tools/event_benchmark/run_eval.py --candidate luna \
  --samples <val_split>/samples.json --media <val_split>/videos \
  --out tools/event_benchmark/results/luna_val24.json

# 2) 测试集打分(50 / 100 / 120-base,已有则复用)

# 3) 标定 + 评估(纯离线,读 score,不调 API)
python -I tools/event_benchmark/tune_threshold.py \
  --val  tools/event_benchmark/results/luna_val24.json \
  --test tools/event_benchmark/results/luna_1fps.json \
  --test tools/event_benchmark/results/luna_100base.json \
  --out  tools/event_benchmark/results/luna_threshold_tuning.json
```

## 小样本注意

每事件留出的片越少,阈值越不稳。现在每类 6 片是下限,报告里要写出每个事件的 val n。某个事件的片少于 6,就把它的 τ 标成 n/a、退回 0.5,别硬报。之后扩了留出集要重跑,并重新核对 `disjoint_assert`。
