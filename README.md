# Ulysses Narrative GMM Turning Points（物語×不確実性×半教師ありGMM）

『Ulysses』を **イベント + 原文引用（evidence）付き**の時系列データへ構造化し、
半教師あり GMM（EM）で **潜在状態（クラスタ）** と **転調点（turning points）** を抽出する。

- **状態クラスタ**：一部のアンカー（labeled）でクラスタの意味を固定
- **境界点（TopK）**：entropy 高 / margin 低（「揺れ」）
- **章ごとの転調密度**：転調の集中度を集計
- **遷移点列の本文還元**：境界点前後のイベント列を抽出して読解へ戻す

> 重要：著作権の扱いは各自の利用条件に従うこと。公開リポジトリでは「短い引用」または「要約」中心を推奨する。

---

## Quickstart（Colab）

下のバッジをクリックして開き、上から実行する。

> ※このリンクは **あなたが GitHub に push した後** に有効になる。
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/Mokafe/Mokafe-ulysses-narrative-gmm/blob/main/notebooks/00_ulysses_gmm_report.ipynb
)


---

## Quickstart（Local）

```bash
pip install -r requirements.txt
python scripts/run_report.py --input data/sample/ulysses_fixed.json --out outputs --plots
# or: python scripts/run_report.py --input data/sample/ulysses_stream.csv --out outputs --plots
```

---

## Outputs（`outputs/`）

- `events_all.csv` : JSONをDataFrame化した全イベント
- `preds_all.csv` : 各点のクラスタ/事後確率/entropy/margin 等
- `boundary_topK.csv` : 境界点TopK（本文へ還元する「揺れ」候補）
- `boundary_context.csv` : 境界点の前後±wイベント（文脈）
- `scene_turning_density.csv` : 章/場面ごとの転調密度
- `scene_cluster_transitions.csv` : 章/場面ごとのクラスタ遷移回数
- `gmm_params.npz` : 学習済みGMMパラメータ（平均・共分散など）
- `fig_*.png` : 図（時系列、転調点、状態遷移など）

---

## Data schema（入力JSONの想定）

`chapter[] -> time_series_data[]` を持つJSONを想定する。各点に以下があれば解析できる。

- `global_step` : 時系列インデックス
- `x` : 特徴ベクトル（例：m, iso など）
- `evidence_en` / `evidence_ja` : 原文引用（または短い抜粋）
- `status` : `labeled` / `unlabeled`
- `label` : アンカー点のみ（例：0/1）

---

## Docs

- Paper（PDF）: `docs/paper/`（公開するなら短い引用・要約中心にすること）

---

## License

MIT
