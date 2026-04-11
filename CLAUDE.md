# keiba-predictor

競馬予測モデル v9b — 桜花賞を主ターゲットとした予測システム。

## アーキテクチャ（3層構造）

```
[Layer 1] ベースNN (src/predictor.py)
    全レース共通。芝レース直近20,000件で学習。μ(能力値)とσ(不確実性)を出力。
        ↓
[Layer 2] コース別QMC (src/qmc_courses.py)
    コース物理特性（直線長、枠順バイアス、脚質補正）を反映した10万回シミュレーション。
    COURSE_PROFILESにコース定義を追加して拡張する。
        ↓
[Layer 3] 議長プロンプト (src/prompts.py)
    3エージェントディベート用プロンプトを自動生成。build_prompt()でデータを埋め込む。
    生成されたプロンプトをLLMに渡して最終推奨5頭を決定。
```

## 使い方（パイプライン）

```python
from src.predictor import Predictor, FEATURES_V9, CAT_FEATURES
from src.qmc_courses import qmc_sim
from src.prompts import build_prompt

pred = Predictor()
pred.train(train_df, ep=50, lr=0.003, seed=42)
nn_preds = pred.predict(race_df)
mc = qmc_sim(nn_preds, race_features=race_df, course='nakayama_turf_1600')
prompt = build_prompt(race_name=..., mc_results=mc, race_features=race_df, nn_preds=nn_preds, ...)
```

## 主要ファイル

| ファイル | 役割 |
|:---|:---|
| `src/predictor.py` | NN定義 + Predictorクラス + 特徴量定義(FEATURES_V9, CAT_FEATURES) |
| `src/qmc_courses.py` | コース別QMC。COURSE_PROFILES辞書にコースを追加して拡張 |
| `src/prompts.py` | 議長プロンプトテンプレート + format_horse_data() + build_prompt() |
| `src/features.py` | v4特徴量構築 (build_all_features) |
| `src/binary_parser.py` | TARGET JV バイナリパーサー (load_all_data, load_hanshin_data) |
| `src/um_parser.py` | 血統データパーサー (load_um_data) |
| `src/sakura_model.py` | 桜花賞専用ドメインスコア（汎用予測では使わない） |
| `scripts/predict_sakura_2026.py` | 2026年桜花賞予測（特徴量構築+v9b学習+QMC） |
| `scripts/predict_saturday_g2.py` | 土曜重賞予測（NZT G2 / 阪神牝馬S G2）のサンプル |
| `scripts/simulate_v9b.py` | バックテスト（2016-2025桜花賞） |

## データ

- `data/features_v9b_cache.pkl` — v9b特徴量キャッシュ（〜2025年）
- `data/features_v9b_2026.pkl` — 2026年込みキャッシュ（〜2026/4/5）
- `data/features_all_v4.pkl` — v4ベース特徴量
- 元データ: `C:/SE_DATA/*.DAT`（TARGET JVバイナリ）
- 出馬表: `C:/TXT/`（TARGET HTMLエクスポート + CSV）

## コースプロファイル（現在定義済み）

- `nakayama_turf_1600` — 中山芝1600m（直線310m、逃げ先行有利）
- `hanshin_turf_1600_outer` — 阪神芝1600m外回り（直線473m、差し追込も届く）

新コース追加時は `src/qmc_courses.py` の COURSE_PROFILES に追加。

## 議長プロンプトのルール

- v9bの数値データのみを根拠とし、外部知識の持ち込み禁止
- Agent A(絶対能力派)、B(舞台適性派)、C(展開リスク派)の3者ディベート
- 最終推奨5頭: 軸馬(1-2頭) + アノマリー枠(必須) + 大穴展開枠(必須)
- 合成の誤謬（安全な妥協案ばかり選ぶこと）を避ける

## v9b モデル仕様

- 入力: 26数値特徴量 + 4カテゴリ埋め込み(騎手/調教師/馬主/種牡馬)
- 構造: Linear(128)→ReLU→Dropout(0.3)→Linear(64)→ReLU→Dropout(0.2)→Linear(32)→ReLU→μ/σ
- 損失: Gaussian NLL
- 学習: 芝レース直近20,000件、50epoch、lr=0.003、seed=42
- バックテスト実績: 馬券内占有率80%（2016-2025桜花賞）
