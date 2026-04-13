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

### 標準ワークフロー（推奨）

出馬表CSV 1枚で Layer 1→2→3 を一気通貫で実行する:

```bash
py -3.13 scripts/predict.py C:/TXT/04112.csv --course fukushima_turf_1200
```

- 入力: TARGET JVエクスポートの出馬表CSV（1行1馬、cp932、ヘッダなし、33列）
- `--course` 省略時は場名+芝ダ+距離から自動検出（未定義なら警告付きでデフォルト実行）
- `--no-debate` で Layer 3 ディベートプロンプトをスキップ可能（非推奨）
- Layer 3 は必ず実行すること。数値出力だけで止めない

### Python API

```python
from src.entry_parser import build_race_features
from src.predictor import Predictor, FEATURES_V9, CAT_FEATURES
from src.qmc_courses import qmc_sim
from src.prompts import build_prompt

feat = pd.read_pickle('data/features_v9b_2026.pkl')
race_info, rf, missing = build_race_features('C:/TXT/04112.csv', feat)
pred = Predictor()
pred.train(train_df, ep=50, lr=0.003, seed=42)
nn_preds = pred.predict(rf)
mc = qmc_sim(nn_preds, race_features=rf, course='fukushima_turf_1200')
prompt = build_prompt(race_name=..., mc_results=mc, race_features=rf, nn_preds=nn_preds, ...)
```

## 主要ファイル

| ファイル | 役割 |
|:---|:---|
| `scripts/predict.py` | **汎用予測スクリプト（推奨エントリーポイント）** CSV1枚でL1→L2→L3一気通貫 |
| `src/entry_parser.py` | **出馬表CSVパーサー** parse_entry_csv() + build_race_features()。騎手乗り替わり時のkisyu_code逆引き修正付き |
| `src/predictor.py` | NN定義 + Predictorクラス + 特徴量定義(FEATURES_V9, CAT_FEATURES) |
| `src/qmc_courses.py` | コース別QMC。COURSE_PROFILES辞書にコースを追加して拡張 |
| `src/prompts.py` | 議長プロンプトテンプレート + format_horse_data() + build_prompt() |
| `src/features.py` | v4特徴量構築 (build_all_features)。kisyu_name, has_same_dist, has_long_stretch含む |
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

- 入力: 30数値特徴量 + 4カテゴリ埋め込み(騎手/調教師/馬主/種牡馬)
- 構造: Linear(128)→ReLU→Dropout(0.3)→Linear(64)→ReLU→Dropout(0.2)→Linear(32)→ReLU→μ/σ
- 損失: Gaussian NLL
- 学習: 芝レース直近20,000件、50epoch、lr=0.003、seed=42
- バックテスト実績: 馬券内占有率80%（2016-2025桜花賞）

## 変更履歴

### 2026-04-11: 騎手コード修正 + 特徴量追加
- **entry_parser.py**: 騎手乗り替わり時のkisyu_code修正を実装。特徴量キャッシュにkisyu_nameが含まれていれば、出馬表CSVの騎手名から逆引きしてkisyu_codeを更新する。キャッシュ再構築が必要。
- **features.py**: build_all_features()のmetaにkisyu_nameを追加。has_long_stretch(直線長コース実績フラグ)を追加。
- **predictor.py**: FEATURES_V9に4特徴量追加: avg_jyuni_1c, avg_jyuni_2c, has_same_dist, has_long_stretch
- **binary_parser.py**: class_cdをRAレコードからパース済み（offset 620, 2桁）
- **注意**: 特徴量キャッシュ(features_v9b_*.pkl)の再構築が必要。kisyu_name列とhas_long_stretch列が追加されるため、既存キャッシュでは騎手コード修正とhas_long_stretch特徴量が機能しない。
