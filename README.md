# Ulysses Narrative GMM Turning Points（物語×不確実性×半教師ありGMM）

『Ulysses』を **イベント + 原文引用（evidence）付き**の時系列データへ構造化し、
半教師あり GMM（EM）で **潜在状態（クラスタ）** と **転調点（turning points）** を抽出する。

本リポジトリは、主たる GMM 転調点分析に加えて、  
**推定された潜在状態ラベルを制御信号として再利用する予備的な SFT 実験メモ** も併載する。

- **状態クラスタ**：一部のアンカー（labeled）でクラスタの意味を固定
- **境界点（TopK）**：entropy 高 / margin 低（「揺れ」）
- **章ごとの転調密度**：転調の集中度を集計
- **遷移点列の本文還元**：境界点前後のイベント列を抽出して読解へ戻す
- **関連実験（予備）**：GMM の潜在状態ラベルを用いた state-conditioned SFT の試行メモ

> 重要：著作権・再配布の扱いは各自の利用条件に従ってください。  
> 本 repo では、研究・検証のための構造化データ、分析コード、報告メモを中心に整理しています。

---

## Reports / Notes

### Main paper (GMM analysis)
- [Paper (PDF): docs/paper/ulysses_imrad_report.pdf](docs/paper/ulysses_imrad_report.pdf)

### Related memo (preliminary SFT experiment)
- [SFT memo (PDF): docs/paper/ulysses_sft_report.pdf](docs/paper/ulysses_sft_report.pdf)

> Note: The SFT material is currently provided as a **preliminary memo / reproducibility note**.  
> GitHub-native notebook integration may be refined later.

---

## Quickstart (Colab)

### GMM report notebook
以下のバッジをクリックすると、Google Colab でレポート（v1.02 修正完了版）を開いて実行できます。

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mokafe/Mokafe-ulysses-narrative-gmm/blob/main/00_ulysses_gmm_report_v1_02.ipynb)

### SFT memo notebook (Drive)
以下は、潜在状態ラベルを利用した SFT 実験メモ用の Colab ノートブックです。  
現時点では GitHub 直リンクではなく、Drive 版を案内しています。

[Open in Colab (Drive)](https://colab.research.google.com/drive/1cdFgKGx3khVEOoRNAsmWL7JfX2VU2GGr)

---

## Quickstart（Local）

```bash
pip install -r requirements.txt
python scripts/run_report.py --input data/sample/ulysses_fixed.json --out outputs --plots
# or: python scripts/run_report.py --input data/sample/ulysses_stream.csv --out outputs --plots
