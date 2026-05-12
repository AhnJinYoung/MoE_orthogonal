# Orthogonalized Expert Aggregation for Sparse MoE Language Models

## 0. One-line Summary

본 연구는 Sparse MoE의 기존 top-k expert aggregation

\[
y = \sum_{i \in \mathrm{TopK}(x)} g_i(x)E_i(x)
\]

대신, **top-1 expert output을 primary direction으로 두고, 나머지 expert output은 top-1 방향과 겹치는 성분을 제거한 뒤 더하는 방식**을 제안한다.

핵심 질문은 다음이다.

> top-k MoE에서 여러 expert를 쓰고 있지만, 실제로는 top-1 expert와 비슷한 방향의 representation을 중복해서 더하고 있는 것은 아닌가?  
> 그렇다면 non-top1 expert가 top1과 orthogonal한 complementary component를 제공하도록 aggregation을 바꾸면 더 효율적인 expert utilization이 가능한가?

---

## 1. Motivation

Sparse MoE는 token마다 일부 expert만 활성화하여 parameter count를 키우면서도 token당 계산량을 제한하는 구조다. 대표적으로 Sparsely-Gated MoE, GShard, Switch Transformer, DeepSeekMoE 등이 있다.

하지만 기존 top-k MoE aggregation은 보통 다음처럼 단순 weighted sum을 사용한다.

\[
y = \sum_{i=1}^{k} g_i v_i
\]

where

\[
v_i = E_i(x)
\]

이때 top-k experts가 모두 서로 다른 정보를 주는 것이 아니라, 서로 비슷한 방향의 output을 반복적으로 더한다면 top-k activation의 효율이 떨어질 수 있다.

본 연구는 **Revisiting Residual Connections: Orthogonal Updates for Stable and Efficient Deep Networks**에서 영감을 받았다. 해당 논문은 residual update에서 module output을 residual stream에 대한 parallel component와 orthogonal component로 나누고, orthogonal component만 더하는 방식을 제안한다.

우리는 이 아이디어를 residual stream이 아니라 **MoE expert aggregation**에 적용한다.

---

## 2. Proposed Method: Top1-Ortho Expert Aggregation

### 2.1 Standard Top-k MoE

기존 방식:

\[
y_{\text{std}} = \sum_{i=1}^{k} g_i v_i
\]

여기서 \(v_1\)은 top-1 expert output, \(v_i\)는 i번째 selected expert output이다.

---

### 2.2 Top1-Ortho Aggregation

top-1 expert output을 anchor로 둔다.

\[
a = v_1
\]

non-top1 expert output에 대해 top1 방향 성분을 제거한다.

\[
v_i^\perp
=
v_i
-
\frac{\langle v_i, a\rangle}{\|a\|^2 + \epsilon}a,
\quad i > 1
\]

최종 MoE output은 다음과 같다.

\[
y_{\text{ortho}}
=
g_1 v_1
+
\sum_{i=2}^{k} g_i v_i^\perp
\]

직관:

- top-1 expert는 primary semantic direction을 담당한다.
- 나머지 experts는 top-1 expert가 이미 설명한 방향을 반복하지 않고, complementary direction만 제공한다.
- 따라서 top-k experts의 marginal utility를 높이는 것이 목표다.

---

## 3. Research Hypotheses

### H1. Pretrained MoE에도 expert redundancy가 존재한다

이미 학습된 MoE 모델에서 top2/top3 expert output이 top1 expert output과 높은 parallel component를 가진다면, 기존 weighted sum은 중복 성분을 포함한다.

---

### H2. Inference-only Top1-Ortho는 일부 pretrained MoE에서 성능을 유지하거나 개선할 수 있다

기존 weight를 바꾸지 않고 aggregation rule만 바꾸었을 때 perplexity나 downstream benchmark가 개선된다면, 이는 pretrained MoE 안에 제거 가능한 redundant expert component가 존재한다는 증거다.

---

### H3. Pretraining부터 Top1-Ortho를 적용하면 expert 구조 자체가 달라진다

inference-only 적용은 기존 모델이 standard sum에 맞춰 학습된 상태에서 억지로 aggregation을 바꾸는 것이다. 반면 pretraining부터 Top1-Ortho를 적용하면 experts가 처음부터 complementary direction을 내도록 적응할 수 있다.

따라서 가장 중요한 비교는 다음이다.

```text
Standard pretrain + Standard inference
vs
Standard pretrain + Top1-Ortho inference
vs
Top1-Ortho pretrain + Top1-Ortho inference
```

---

## 4. Experimental Setup

## 4.1 Large Pretrained MoE Inference Experiments

기존 pretrained MoE 모델에 대해 weight는 고정하고 MoE aggregation만 교체한다.

### Target Models

```text
1. Gemma 4 26B
2. gpt-oss-20B
3. Qwen 3.6 35B
4. Nemotron 3 Nano
```

목표는 대형 모델을 학습하는 것이 아니라, **이미 학습된 MoE에서 inference-time orthogonal aggregation이 작동하는지** 확인하는 것이다.

### Conditions

| Condition | Pretraining | Inference aggregation |
|---|---|---|
| Standard | original | standard top-k sum |
| Inference-only Top1-Ortho | original | Top1-Ortho |
| Inference-only GS-Ortho | original | Gram-Schmidt Ortho |
| Inference-only Res-Ortho | original | Residual-stream Ortho |

이 실험에서는 너무 많은 variant를 넣지 않는다. 핵심은 다음 세 개다.

```text
1. Standard
2. Top1-Ortho
3. GS-Ortho or Res-Ortho
```

GS-Ortho와 Res-Ortho 중 하나만 남겨도 된다. 개인적으로는 reviewer 설득용으로 둘 다 있으면 좋지만, 비용이 부담되면 `Top1-Ortho`와 `Res-Ortho`만 남기는 것이 더 깔끔하다.

---

## 4.2 Small-scale MoE Pretraining Experiments

대형 모델은 pretraining부터 적용하기 어렵기 때문에, 별도의 500M~1B급 vanilla MoE 모델을 학습한다.

### Model Config

DeepSeekMoE 스타일의 vanilla MoE config를 기반으로 한다. DeepSeekMoE는 expert specialization 문제를 직접 다룬 대표적인 MoE 구조이며, routed experts와 shared experts를 통해 expert redundancy를 줄이려는 방향을 제안한다.

단, 본 실험에서는 architecture novelty를 최소화하기 위해 복잡한 DeepSeekMoE full design을 그대로 따라가기보다는, 다음과 같은 단순 MoE Transformer를 사용한다.

```yaml
model:
  type: decoder-only Transformer MoE
  size: 500M ~ 1B total parameters
  hidden_size: 1024 or 1536
  num_layers: 16 ~ 24
  num_attention_heads: 16
  num_experts: 8 or 16
  top_k: 2
  moe_layer_frequency: every 2 layers or every layer
  expert_type: SwiGLU FFN
  router: learned linear router + softmax top-k
  auxiliary_loss: standard load balancing loss
  tokenizer: Llama/Qwen-compatible tokenizer or custom BPE
```

핵심은 **architecture를 최대한 평범하게 두고 aggregation만 바꾸는 것**이다.

---

## 4.3 Pretraining Data

주 데이터셋은 FineWeb-Edu sample-100BT를 사용한다. Hugging Face의 FineWeb-Edu는 full dataset 외에 `sample-10BT`, `sample-100BT`, `sample-350BT` config를 제공하며, `fineweb_edu_100BT`는 full FineWeb-Edu 약 1.6T tokens에서 fixed seed로 sampling한 약 100B token subset으로 설명되어 있다.

대안 또는 sanity check로 WikiText-103을 사용한다. WikiText-103은 약 103M tokens의 long-form Wikipedia language modeling dataset으로, 작은 scale에서 빠른 PPL 비교에 적합하다.

### Recommended Data Plan

```text
Primary:
  - FineWeb-Edu sample-100BT

Fast sanity check:
  - WikiText-103

Optional:
  - FineWeb-Edu sample-10BT for pilot
```

---

## 4.4 Pretraining Conditions

pretraining은 총 두 번만 한다.

| Model ID | Training aggregation | Inference aggregation | Purpose |
|---|---|---|---|
| A | Standard top-k sum | Standard top-k sum | vanilla MoE baseline |
| B | Standard top-k sum | Top1-Ortho | inference-only effect |
| C | Top1-Ortho | Top1-Ortho | proposed full method |

중요한 점:

```text
A와 B는 같은 checkpoint를 사용한다.
즉 Standard MoE는 한 번만 pretrain하고,
eval할 때 standard aggregation과 Top1-Ortho aggregation을 둘 다 적용한다.
```

따라서 실제 pretraining run은 두 개뿐이다.

```text
Run 1: Standard MoE pretraining
Run 2: Top1-Ortho MoE pretraining
```

---

## 4.5 Training Budget

A100 40GB 4장 기준으로 현실적인 budget을 다음처럼 둔다.

```text
Pilot:
  - 500M MoE
  - FineWeb-Edu 1B ~ 5B tokens
  - 목적: 구현 검증, loss curve 확인

Main:
  - 500M ~ 1B MoE
  - FineWeb-Edu sample-100BT 중 20B ~ 100B tokens
  - 목적: 논문 main result

Fallback:
  - WikiText-103 or FineWeb-Edu 10BT
  - 목적: 빠른 ablation 및 PPL sanity check
```

100B tokens를 전부 학습하면 가장 깔끔하지만, 4×A100 40GB에서는 시간이 꽤 걸릴 수 있으므로, 논문 작성용으로는 다음을 권장한다.

```text
Main claim:
  20B ~ 50B tokens

Extra scaling result:
  100B tokens if feasible
```

---

# 5. Evaluation

## 5.1 Main Performance Metrics

### Language Modeling

| Dataset | Metric |
|---|---|
| FineWeb-Edu validation split | loss, perplexity |
| WikiText-103 | perplexity |
| C4 validation subset | loss, perplexity |

### Downstream Benchmarks

coding benchmark와 long-context retrieval benchmark는 제외한다.

| Benchmark | Metric |
|---|---|
| ARC-Challenge | accuracy |
| HellaSwag | accuracy |
| PIQA | accuracy |
| WinoGrande | accuracy |
| MMLU subset | accuracy |
| GSM8K subset | exact match |

평가는 EleutherAI lm-evaluation-harness를 사용하는 것이 좋다. lm-eval은 language model evaluation을 재현 가능하고 확장 가능하게 수행하기 위한 open-source evaluation harness로 소개되어 있다.

---

# 6. Core Structural Metrics

기존 benchmark 성능만으로는 “그냥 운 좋게 PPL이 좋아진 것”인지, “expert 구조가 실제로 좋아진 것”인지 알기 어렵다. 따라서 본 논문에서는 structural metric을 최소한으로 유지하되, 핵심적인 것만 남긴다.

---

## 6.1 Expert Parallel Component Ratio, PCR

가장 중요한 metric이다.

non-top1 expert output이 top1 expert output 방향과 얼마나 겹치는지 측정한다.

\[
\mathrm{PCR}_i
=
\frac{
\|\mathrm{proj}_{v_1}(v_i)\|_2
}{
\|v_i\|_2 + \epsilon
}
\]

전체 평균:

\[
\mathrm{PCR}
=
\mathbb{E}_{l,t,i>1}
\left[
\frac{
\|\mathrm{proj}_{v_1}(v_i)\|_2
}{
\|v_i\|_2 + \epsilon
}
\right]
\]

해석:

```text
높은 PCR:
  non-top1 expert가 top1과 중복된 방향을 많이 냄.

낮은 PCR:
  non-top1 expert가 top1과 다른 방향을 많이 냄.
```

보고할 것:

```text
1. PCR over training
2. PCR per layer
3. PCR: Standard-pretrained vs Top1-Ortho-pretrained
```

가장 중요한 expected result:

```text
Standard pretraining:
  PCR이 높게 유지됨.

Top1-Ortho pretraining:
  projection을 적용하기 전의 raw expert outputs에서도 PCR이 낮아짐.
```

이 결과가 나오면 “우리 방식이 단순 inference trick이 아니라 expert geometry 자체를 바꾼다”는 주장을 할 수 있다.

---

## 6.2 Removed Redundancy Energy, RRE

Top1-Ortho가 제거한 성분의 크기를 측정한다.

기존 standard output:

\[
y_{\text{std}}
=
\sum_i g_i v_i
\]

Top1-Ortho output:

\[
y_{\text{ortho}}
=
g_1v_1
+
\sum_{i>1}g_i v_i^\perp
\]

제거된 성분:

\[
y_{\text{removed}}
=
y_{\text{std}} - y_{\text{ortho}}
\]

Removed Redundancy Energy:

\[
\mathrm{RRE}
=
\frac{
\|y_{\text{removed}}\|_2
}{
\|y_{\text{std}}\|_2 + \epsilon
}
\]

해석:

```text
RRE가 높다:
  기존 MoE output 중 top1 방향으로 중복해서 더해진 성분이 크다.

RRE가 낮다:
  기존 MoE도 이미 complementary expert outputs를 만들고 있다.
```

보고할 것:

```text
1. pretrained large MoE에서 RRE
2. small pretraining에서 RRE over training
3. layer-wise RRE
```

---

## 6.3 Effective Non-top1 Contribution, ENC

top2/top3 expert가 실제 loss에 기여하는지 측정한다.

\[
\Delta L_{\text{non-top1}}
=
L(y \text{ without non-top1 experts}) - L(y)
\]

또는 top2만 제거:

\[
\Delta L_{\text{top2}}
=
L(y \text{ without top2 expert}) - L(y)
\]

해석:

```text
ENC가 낮다:
  top2/top3 expert를 써도 실제 loss 개선이 작음.
  즉 top-k를 nominally 쓰지만 effective contribution은 작음.

ENC가 높다:
  non-top1 expert가 실제로 예측에 중요한 정보를 제공함.
```

기대 결과:

```text
Top1-Ortho pretraining에서는 non-top1 expert 제거 시 loss 증가가 더 커져야 한다.
즉 non-top1 expert의 marginal utility가 커져야 한다.
```

---

# 7. Diagnostic Metrics

논문 분량과 실험 부담을 고려해서 diagnostic metric은 2개만 남긴다.

---

## 7.1 Expert Output Cosine Similarity

top-k expert output 사이의 평균 cosine similarity를 측정한다.

\[
\mathrm{CosSim}
=
\mathbb{E}_{i \neq j}
[
\cos(v_i, v_j)
]
\]

보고할 것:

```text
1. layer-wise expert output cosine similarity
2. training step별 cosine similarity
```

기대 결과:

```text
Top1-Ortho pretraining은 Standard pretraining보다 expert output similarity가 낮아야 한다.
```

---

## 7.2 Load Balance

orthogonal aggregation이 routing collapse를 유발하지 않는지 확인한다.

각 expert가 받은 token 비율:

\[
Load_i
=
\frac{
\# \text{tokens routed to expert } i
}{
\# \text{total routed tokens}
}
\]

보고 metric:

```text
1. expert load variance
2. max/min load ratio
3. dropped token ratio if capacity limit exists
```

해석:

```text
성능이 좋아져도 load balance가 무너지면 나쁜 결과다.
Top1-Ortho는 aggregation만 바꾸므로 router 자체를 직접 바꾸지는 않지만,
pretraining dynamics를 통해 routing distribution이 달라질 수 있다.
```

---

# 8. Orthogonalization Variants

실험 variant는 최소화한다.

## 8.1 Main Method: Top1-Ortho

\[
v_i^\perp
=
v_i
-
\mathrm{proj}_{v_1}(v_i)
\]

\[
y
=
g_1v_1
+
\sum_{i>1}g_i v_i^\perp
\]

이 방법이 본 논문의 proposed method다.

---

## 8.2 GS-Ortho

top-k experts를 순서대로 Gram-Schmidt orthogonalization한다.

\[
u_1 = v_1
\]

\[
u_i
=
v_i
-
\sum_{j<i}
\mathrm{proj}_{u_j}(v_i)
\]

\[
y
=
\sum_i g_i u_i
\]

목적:

```text
Top1-Ortho가 충분한지,
아니면 top-k 전체를 서로 orthogonalize해야 하는지 비교한다.
```

단점:

```text
1. top-k ordering에 민감함
2. k가 커질수록 계산량 증가
3. numerical instability 가능성
```

따라서 main method가 아니라 ablation이다.

---

## 8.3 Res-Ortho

MoE output 전체를 residual stream에 대해 orthogonalize한다.

\[
m
=
\sum_i g_i v_i
\]

\[
m^\perp
=
m
-
\mathrm{proj}_{x}(m)
\]

\[
h_{l+1}
=
h_l
+
m^\perp
\]

목적:

```text
이 방법은 원래 Orthogonal Residual Update 논문과 가장 가까운 baseline이다.
우리 방식이 residual stream orthogonalization보다 MoE-specific하게 더 좋은지 확인한다.
```

---

# 9. Main Tables

## 9.1 Large Pretrained MoE Inference Table

| Model | Aggregation | PPL ↓ | ARC-C ↑ | HellaSwag ↑ | PIQA ↑ | MMLU subset ↑ | GSM8K subset ↑ | PCR ↓ | RRE ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Gemma 4 26B | Standard |  |  |  |  |  |  |  |  |
| Gemma 4 26B | Top1-Ortho |  |  |  |  |  |  |  |  |
| Gemma 4 26B | Res-Ortho |  |  |  |  |  |  |  |  |
| gpt-oss-20B | Standard |  |  |  |  |  |  |  |  |
| gpt-oss-20B | Top1-Ortho |  |  |  |  |  |  |  |  |
| Qwen 3.6 35B | Standard |  |  |  |  |  |  |  |  |
| Qwen 3.6 35B | Top1-Ortho |  |  |  |  |  |  |  |  |
| Nemotron 3 Nano | Standard |  |  |  |  |  |  |  |  |
| Nemotron 3 Nano | Top1-Ortho |  |  |  |  |  |  |  |  |

---

## 9.2 Small MoE Pretraining Table

| Training Run | Train Aggregation | Eval Aggregation | Val Loss ↓ | PPL ↓ | ARC-C ↑ | HellaSwag ↑ | PIQA ↑ | PCR ↓ | RRE ↓ | ENC ↑ |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Run 1 | Standard | Standard |  |  |  |  |  |  |  |  |
| Run 1 | Standard | Top1-Ortho |  |  |  |  |  |  |  |  |
| Run 2 | Top1-Ortho | Top1-Ortho |  |  |  |  |  |  |  |  |

이 table이 논문의 핵심이다.

---

## 9.3 Variant Ablation Table

| Method | Val Loss ↓ | PPL ↓ | PCR ↓ | RRE ↓ | ENC ↑ | Load Balance |
|---|---:|---:|---:|---:|---:|---:|
| Standard |  |  |  |  |  |  |
| Top1-Ortho |  |  |  |  |  |  |
| GS-Ortho |  |  |  |  |  |  |
| Res-Ortho |  |  |  |  |  |  |

---

# 10. Essential Figures

## Figure 1. Method Diagram

```text
Token representation x
  → Router
  → top-k experts
  → keep top1 output
  → project non-top1 outputs orthogonal to top1
  → gated aggregation
  → residual update
```

---

## Figure 2. Training Loss Curve

```text
x-axis: training tokens
y-axis: validation loss

Curves:
  1. Standard train / Standard eval
  2. Standard train / Top1-Ortho eval
  3. Top1-Ortho train / Top1-Ortho eval
```

---

## Figure 3. PCR over Training

```text
x-axis: training tokens
y-axis: Expert Parallel Component Ratio

Curves:
  1. Standard pretraining
  2. Top1-Ortho pretraining
```

핵심 메시지:

```text
Top1-Ortho pretraining은 expert output 자체의 redundancy를 줄인다.
```

---

## Figure 4. Effective Non-top1 Contribution

```text
x-axis: model condition
y-axis: loss increase when removing non-top1 experts
```

핵심 메시지:

```text
Top1-Ortho pretraining에서는 non-top1 expert가 실제로 더 중요한 역할을 한다.
```

---

# 11. Expected Results and Interpretation

## Case A. Inference-only Top1-Ortho improves pretrained models

강한 결과다.

해석:

```text
Pretrained MoE already contains redundant expert output components.
Removing top1-parallel components at inference time improves aggregation.
```

논문 claim:

```text
Top1-Ortho is a drop-in inference-time aggregation method for pretrained MoE models.
```

---

## Case B. Inference-only는 별로지만 pretraining부터 하면 좋아짐

가장 그럴듯하고 논문으로도 좋은 결과다.

해석:

```text
Standard MoE models are calibrated for standard summation,
so inference-only orthogonalization may disturb learned representations.
However, if trained from scratch with Top1-Ortho, experts adapt to become complementary.
```

논문 claim:

```text
Orthogonalized aggregation changes expert learning dynamics and yields more efficient expert specialization.
```

---

## Case C. Benchmark gain은 작지만 structural metric이 좋아짐

그래도 논문화 가능성은 있다.

해석:

```text
Top1-Ortho reduces expert redundancy and increases non-top1 expert utility,
but performance gains may require larger scale or better tuning.
```

이 경우 논문은 method paper보다는 analysis + architecture proposal 성격이 강해진다.

---

## Case D. 성능이 떨어짐

해석:

```text
Parallel expert components may not be merely redundant.
They may act as calibrated ensemble-style reinforcement.
Removing them breaks scale and feature composition.
```

이 경우 추가로 projection strength를 도입할 수 있다.

\[
v_i^{\perp,\alpha}
=
v_i
-
\alpha \cdot \mathrm{proj}_{v_1}(v_i)
\]

\[
\alpha \in [0, 1]
\]

하지만 1차 proposal에서는 \(\alpha\) ablation을 optional로 둔다.

---

# 12. Minimal Implementation Plan

## Phase 1. Large MoE Hooking

```text
1. pretrained MoE model 로드
2. MoE layer에서 router logits, top-k indices, gate weights, expert outputs 추출
3. original standard aggregation을 hook으로 재현
4. Top1-Ortho aggregation으로 교체
5. PPL부터 측정
6. PCR, RRE 계산
```

성공 기준:

```text
hook으로 standard aggregation을 재현했을 때 original model output과 거의 동일해야 한다.
```

---

## Phase 2. Small MoE Pretraining

```text
1. 500M급 vanilla MoE 구현
2. FineWeb-Edu sample-100BT dataloader 구성
3. Run 1: Standard MoE pretraining
4. Run 2: Top1-Ortho MoE pretraining
5. 동일 validation set에서 PPL/loss 비교
6. 동일 benchmark subset 평가
7. PCR, RRE, ENC 계산
```

---

## Phase 3. Paper Analysis

```text
1. benchmark table
2. loss curve
3. PCR curve
4. RRE layer-wise plot
5. ENC comparison
6. load balance sanity check
```

---

# 13. Required References

## 13.1 Core MoE Papers

1. **Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer**  
   원조 sparse MoE 계열. trainable gating network로 sparse expert combination을 수행하는 기본 구조를 제안했다.  
   Link: https://arxiv.org/abs/1701.06538

2. **GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding**  
   Transformer 기반 대규모 MoE와 top-k routing, sharding을 다룬 대표 논문.  
   Link: https://arxiv.org/abs/2006.16668

3. **Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity**  
   top-1 routing을 중심으로 MoE를 단순화하고 bf16 training 안정성을 보여준 논문.  
   Link: https://arxiv.org/abs/2101.03961

4. **Mixture-of-Experts with Expert Choice Routing**  
   token이 expert를 고르는 대신 expert가 token을 고르는 routing을 제안한다. routing 방식 비교 reference로 필요하다.  
   Link: https://arxiv.org/abs/2202.09368

5. **DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models**  
   expert specialization과 routed/shared expert 설계를 다루므로, 본 연구의 small MoE config 및 expert redundancy discussion에 중요하다.  
   Link: https://arxiv.org/abs/2401.06066

---

## 13.2 Orthogonality / Expert Diversity Papers

6. **Revisiting Residual Connections: Orthogonal Updates for Stable and Efficient Deep Networks**  
   본 연구의 직접적인 inspiration. residual update에서 parallel component를 제거하고 orthogonal component만 더하는 방식.  
   Link: https://arxiv.org/abs/2505.11881

7. **Diversifying the Mixture-of-Experts Representation for Language Models with Orthogonal Optimizer**  
   OMoE. expert representation similarity 문제를 지적하고 orthogonal expert optimizer를 제안한다. 본 연구와 가장 가까운 orthogonal MoE related work 중 하나다.  
   Link: https://arxiv.org/abs/2310.09762

8. **Advancing Expert Specialization for Better MoE**  
   load balancing loss가 expert overlap과 overly uniform routing을 만들 수 있다고 보고, orthogonality loss와 variance loss를 추가한다. 본 연구와 달리 aggregation operator가 아니라 loss-level method다.  
   Link: https://arxiv.org/abs/2505.22323

9. **On the Representation Collapse of Sparse Mixture of Experts**  
   Sparse MoE의 routing이 token representation collapse를 유도할 수 있음을 분석한다. expert/output redundancy discussion에 필요하다.  
   Link: https://openreview.net/forum?id=mWaYC6CZf5

---

## 13.3 Alternative MoE Aggregation / Routing Papers

10. **From Sparse to Soft Mixtures of Experts**  
    Sparse top-k routing 대신 differentiable soft assignment를 사용하는 Soft MoE를 제안한다. “expert aggregation/routing을 바꾸는 대안적 MoE” reference로 필요하다.  
    Link: https://arxiv.org/abs/2308.00951

11. **Sparse MoE as the New Dropout: Scaling Dense and Self-Slimmable Transformers**  
    activated expert 수를 점진적으로 늘리는 SMoE-Dropout을 제안한다. MoE training dynamics와 collapse discussion에 보조 reference로 쓸 수 있다.  
    Link: https://openreview.net/forum?id=w1hwFUb_81

---

## 13.4 Dataset / Evaluation References

12. **FineWeb-Edu / FineWeb-Edu sample-100BT**  
    본 pretraining의 primary dataset. Hugging Face dataset card 기준 sample-100BT는 full FineWeb-Edu에서 sampling한 약 100B token subset이다.  
    Link: https://huggingface.co/datasets/HuggingFaceFW/fineweb_edu_100BT

13. **WikiText-103**  
    빠른 sanity-check용 LM dataset. 약 103M tokens의 Wikipedia long-form language modeling dataset이다.  
    Link: https://huggingface.co/datasets/Salesforce/wikitext

14. **Language Model Evaluation Harness**  
    downstream benchmark evaluation을 위한 reproducible evaluation framework.  
    Link: https://github.com/EleutherAI/lm-evaluation-harness

15. **TinyLlama: An Open-Source Small Language Model**  
    1.1B급 small LLM pretraining의 참고 사례. TinyLlama는 1.1B 모델을 약 1T tokens로 pretrain한 사례라, small-scale LLM training setup reference로 유용하다.  
    Link: https://arxiv.org/abs/2401.02385

---

# 14. Final Paper Claim Draft

본 논문에서 주장할 수 있는 가장 안전한 claim은 다음이다.

> We propose Top1-Ortho, a simple modification to sparse MoE expert aggregation. Instead of directly summing all selected expert outputs, we preserve the top-1 expert output and add only the components of non-top1 expert outputs that are orthogonal to the top-1 direction. Unlike prior work that encourages expert diversity through routing objectives, auxiliary losses, or orthogonal optimization, our method directly modifies the forward aggregation operator. Through pretrained MoE inference experiments and controlled 500M–1B MoE pretraining experiments, we evaluate whether orthogonalized aggregation reduces expert redundancy, increases effective non-top1 expert contribution, and improves language modeling performance.

---

# 15. Priority Checklist

가장 먼저 해야 할 순서:

```text
1. 500M vanilla MoE config 확정
2. Standard MoE와 Top1-Ortho MoE forward 구현
3. Standard aggregation hook 재현 테스트
4. WikiText-103 또는 FineWeb-Edu small subset으로 pilot
5. FineWeb-Edu sample-100BT로 Run 1: Standard pretraining
6. 같은 checkpoint로 Standard eval / Top1-Ortho eval 둘 다 수행
7. Run 2: Top1-Ortho pretraining
8. PPL, loss, PCR, RRE, ENC 측정
9. large pretrained MoE에서 inference-only Top1-Ortho 적용
10. paper table/figure 작성
```
