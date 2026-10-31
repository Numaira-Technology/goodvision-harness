# 阈值标定协议(SPEC §5/§8 展开)

## 为什么 0.5 对 decision/threshold 类不公平

片级判定 = **帧分数取最大 ≥ gate**。

- **language 类**(qwen/gemini):每帧输出被压成硬 **0/1**(JSON yes/no)。gate 0.5 等价于"任一帧 true",
  这是个**离散**判定,0.5 之外没有可调空间 —— 调它没有意义。
- **decision 类**(luna `noul`)/ **threshold 类**(CLIP/YOLO logit):每帧是一个**连续校准分数**。
  0.5 只是从二值逻辑借来的边界,**未必是这个模型的真实工作点**。合适的阈值要靠数据标定。

"取最大"还是把双刃剑:多抽帧对正例是多机会抓到事件(召回↑),对负例是多机会误触发(误报↑)。
一个每帧误报偏高的模型(如 luna 对 fall)在 0.5 下误报会被放大 —— **抬高该事件阈值**正是对症的修法。

## 唯一正当的做法:留出集选阈 → 冻结 → 测试集评估

1. **留出集(held-out validation)**:一组**不在任何测试集里**的片子(`validation_split.json`,
   `disjoint_assert` 全 0)。只在这上面选阈。
2. **按事件网格扫阈**:τ∈[0,1] 每 0.01,算验证集准确率,取**并列最优区间的中点**(max-margin,最稳)。
   `tune_threshold.py` 的 `pick_tau` 就是这个。
3. **冻结**选出的 per-class τ。
4. **测试集评估**:用冻结的 τ 在测试集上算分,与 **@0.5 基线**并列报告。
5. **主表永远是 0.5**。校准结果是脚注/变体,**绝不替换主表**,也不进 Drive 冻结集的主结论。

违反任何一条(尤其"在测试集上选阈")= 过拟合测试集,数字作废。

## 为什么不能在测试集上选阈

在测试集上挑能最大化测试分数的阈值,等于把测试集答案偷看进了"模型配置",报出来的分数是虚高的、
不可推广的。留出集的意义就是"选阈用的数据"和"报分用的数据"严格分开。

## worked example(luna,v1)

留出集 24 片(fight/smoking/fall/weapon 各 6,12 正 12 负,1fps),与 50/100 测试集零重叠。

选出的 per-class τ:

| 事件 | τ | 相对 0.5 | 作用 |
|---|---|---|---|
| fight | 0.29 | 调低 | 多抓打斗(召回↑) |
| smoking | 0.53 | ≈0.5 | 基本不变 |
| fall | 0.58 | 调高 | 砍掉把日常动作当跌倒的误报 |
| weapon | 0.46 | ≈0.5 | 基本不变 |

测试集结果(@冻结 τ vs @0.5):

| test | @0.5 | @tuned |
|---|---|---|
| 50 片 | 84% | **94%** |
| 100 题 | 85% | **92%** |

提升几乎全来自 **fall**:0.5 下 luna fall 误报多(50 片 FP4、100 题 FP7),阈值抬到 0.58 后这些误报消失。

## 命令

```sh
# 1) 留出集打分(若已有 results/luna_val24.json 且与冻结 split 一致,跳过)
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

每事件留出片越少,阈值越毛糙。当前每类 6 片是下限;报告里务必写出每事件的 val n。
某事件 val 片 < ~6 → 该事件 τ 标 `n/a`、退回 0.5,别硬报。扩充留出集后重跑、并复核 `disjoint_assert`。
