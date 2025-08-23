# CPC‑MS に基づく Multi‑Agent Orchestration Demo（Jupyter Notebook 解説）

**想定読者**：生成モデルや深層学習の数理に詳しくない方。ただし、確率・統計や機械学習の基本語彙（確率、尤度、相互情報量、回帰など）は登場します。
**目的**：Notebook が何をしているか、式は最小限で丁寧に説明します。数式の由来は Collective Predictive Coding（CPC）と CPC‑MS（Science モデル）にあります。理論的背景は 図と式を本文中に参照します（ページ番号つき）。

⸻

## 1. 全体像（TL;DR）
- 各データソース（X/Twitter, YouTube, TikTok, 新聞, テレビ）ごとに専門家エージェントが、時系列の出来事（投稿/記事/発言）のカウントから内部表象 $z^k(t)$ を推定します（per‑source inference）。
- 複数エージェントの $z^k(t)$ を束ねて、社会的に共有される外的表象（＝語彙やコードブック） $w$ を分散に推定します。
数式で要点だけ言えば、最適事後は
\[
q^\*(w \mid z)\ \propto\ p(w)\,\{p(z\mid w)\}^{1/\lambda}
\]
で与え られ、$\lambda$ が小さいほど one‑hot に近いハード割当になります（理論の導出と図解は 「新しい通信の数学的理論」 p.5–7 を参照）。 ￼
- Supervisor + Network（LangGraph のワークフロー）で、
  ① 縦の流れ：o_k \to z_k \to w（推論）と、
  ② 逆向き：w \to z_k \to o_k（生成の整合性チェック：負の対数尤度/NLLと分布距離/JS）
を回しながら、横連携で MH ネーミングゲーム（MHNG）の提案・受理を行い、共有語彙 w を分散更新します（受理率 \alpha=\min(1,\ p(z\mid w’)/p(z\mid w))）。MHNG と科学活動の対応は CPC‑MS 図2（p.4）を参照。 ￼
- 因果の仮説は、ソース間の遅れと媒介をもつ有向グラフとして推定します（例：TikTok → w → Twitter）。**総効果＝直接＋間接（媒介）**に分解します。
- Active Inference：将来データに対する期待自由エネルギー G(\tilde a) を小さくする行動（小さな実験計画）を玩具実装します（式(2.8)–(2.10)，CPC‑MS p.8–9）。 ￼

**理論的帰結**：CPC では、個の表象 z と 集団の外的表象 w を同時に最適化するための到達可能性境界
H(z,o) - D(z,o\mid w) \ \le\ I(w;z)\ \le\ C(w;z)
（Collective Rate–Distortion）が成り立ちます（新しい通信の数学的理論 式(23), p.4–5）。Notebook はこの観点で「予測誤差」と「集団正則化」のバランスを可視化します。 ￼

⸻

## 2. 変数と対応（用語の最小セット）
- `o_k(t)`：ソース k の観測（例：支持/批判/中立の件数など、時系列イベント）。
- `z_k(t)`：ソース k の内部表象（3次元ベクトル：支持/批判/中立の内部強度）。
- `w`\in\{1,\dots,K\}：共有語彙（話題コード）のラベル。
- **コードブック**：各コード w に対する中心 \mu_w（\mathbb{R}^3）と事前 p(w)。

**CPC‑MS の PGM**：図3（p.5）に、w（上段）→ z_k（中段）→ o_k（下段）という生成の向き（上→下）と、推論の向き（下→上）が描かれています。本Notebookはこの因子化
p(w,z,o) = p(w)\prod_k p(z_k\mid w)\,p(o_k\mid z_k)
に沿って実装しています。 ￼

⸻

## 3. Notebook の処理フロー（セルごとの意味）

### 3.1 ダミーデータ生成（events → o）
- 各ソースのタイムラインに「支持・批判・中立」のカウントを少量生成し、基礎的な遅れ相関（例：TikTok が先、Twitter が後）を仕込みます。
- これが観測 `o_k(t)` です。

### 3.2 専門家エージェント：o → z（各ソースの内部表象）
- 簡易に、正規化・平滑化・ロジット変換などで `o_k(t)` を \mathbb{R}^3 の連続表現 `z_k(t)` に写像します（「支持・批判・中立」の連続強度）。
- CPC の個別推論に相当：\,q(z_k\mid o_k)（CPC‑MS 式(2.2) 右側の個別項）。 ￼

### 3.3 共有語彙（コードブック）学習：z → w（CPC の核）
- **E‑step**（posterior）：
\[
q^\*(w\mid z)\ \propto\ p(w)\,\{p(z\mid w)\}^{1/\lambda}
\]
をソフト割当（\lambda>0）で計算し、\lambda\!\downarrow\!0 で ハード割当に近づきます（新しい通信の数学的理論 式(29)–(36)，p.5–7 と図1/表1 p.7 の直観参照）。 ￼
- **M‑step**（codebook 更新）：ソフト責務で各中心 \mu_w を更新。
- **ダッシュボード**：`NLL(o|z)`、`NLL(z|w)`、JS 距離などを可視化し、生成の整合性 w\to z\to o を検証。

**CPC の自由エネルギー分解**（CPC‑MS 式(2.7), p.8–9）：
\underbrace{\mathbb{E}[\ln q(w\mid z)]}{\text{collective}}\
-\ \sum_k \underbrace{\mathbb{E}[\ln p(o_k\mid z_k)]}{\text{prediction}}\
-\ \sum_k \underbrace{\mathbb{E}\!\left[\ln \frac{p(z_k\mid w)}{q(z_k\mid o_k)}\right]}_{\text{individual}}
Notebook のメトリクスは、この三項のバランスの目安になります。 ￼

### 3.4 因果グラフ（時間遅れ＋媒介）
- **総効果**（例：TikTok \to Twitter）を「遅れ特徴」を使った回帰で見積もり、
- **媒介効果**：TikTok \to w \to Twitter に分解（二段推定）。総効果＝直接＋媒介を別掲します。
- これは w を媒介変数として扱い、単一ソースでは見えない相互作用を解像します。

### 3.5 分散推論（MHNG、一回ラウンド）
- 各ソースが話者として w’ を提案し、聞き手が
\alpha = \min\!\left(1,\ \frac{p(z_{\text{listener}}\mid w’)}{p(z_{\text{listener}}\mid w)}\right)
で受理判定。これが分散ベイズ推論としての MH ネーミングゲームのコアです（図2 p.4、付録B p.20 参照）。 ￼
- Notebook では、受理率と通信ログを出力します。

### 3.6 Active Inference（実験計画の玩具）
- 将来行動 \tilde a ごとの期待自由エネルギー
G(\tilde a)=\mathbb{E}\!\left[\ln q(\tilde w\mid \{\tilde z_k\})\right]
-\sum_k \mathbb{E}\!\left[\ln p(\tilde o_k\mid \tilde z_k,\tilde a_k,\tilde C_k)\right]
-\sum_k \mathbb{E}\!\left[\ln \frac{p(\tilde z_k\mid \tilde w,\tilde a_k)}{q(\tilde z_k\mid \tilde o_k,\tilde a_k)}\right]
を小さくする簡便戦略を選びます（式(2.8)–(2.10) の簡約）。探索（知識獲得）と活用（誤差減少）のトレードオフを直観できます。 ￼

⸻

## 4. 数式の由来（図とページで直観をつかむ）
### 1. コードブック創発

- **新しい通信の数学的理論 §7「コードブックの創発」**：
\[
q^\*(w\mid z)\propto p(w)\{p(z\mid w)\}^{1/\lambda}
\]
から \lambda\downarrow 0 で one‑hot 割当（図と表は p.7）。Notebook の E/M 更新はこの式をそのまま模倣した玩具版です。 ￼

### 2. CRD 不等式（集団レート歪み）

- **同上 §5–6**：
H(z,o)-D(z,o\mid w)\le I(w;z)\le C(w;z)
は、個の誤差 D と 集団の情報 I,C の到達可能性を与えます。Notebook の NLL/JS は、この境界に近づくよう整合性を点検する意図です（図は p.4–5）。 ￼

### 3. CPC‑MS の生成・推論分解

- **CPC‑MS §2.3（図3 p.5、式(2.1)–(2.7) p.8–9）**：生成 p(w)p(z\mid w)p(o\mid z) と、推論 q(z\mid o), q(w\mid z) の分解。Notebook の縦方向（推論）と逆向き（生成整合）はこの図を忠実に反映。 ￼

### 4. MH ネーミングゲーム（分散ベイズ）

- **CPC‑MS 図2 p.4・付録B（p.20）**：言語ゲーム \Leftrightarrow 分散 MH サンプラー。Notebook の受理率 \alpha はこの式型です。 ￼

⸻

## 5. 出力の読み方（最低限の指針）
- **Codebook centers (\mu_w)**：各話題コードの原型ベクトル。
- **assign_soft/assign_hard**：時刻ごとの語彙割当（ソフト/ハード）。ハードの多数派がそのソースの「現在の話題」。
- **NLL(o|z), NLL(z|w)**：生成整合性。小さいほど良い。
- **JS 距離**：各分布の似てなさ。小さいほど整合。
- **Causal effects**：総効果・直接・媒介を別掲（媒介が大きい＝共有語彙 w を通じた影響が強い）。
- **MHNG acceptance**：分散更新の合意のしやすさ。低すぎる場合はコードブックが不整合（センターが遠い）。

⸻

## 6. なぜ単一ソースでは見抜けない因果が見えるのか？
- 単一ソースだと、同一プラットフォーム内の「流行の入れ替わり」は追えても、「どこから伝播したのか」は観測できません。
- CPC では、外的表象 w（共有語彙）を媒介として明示化します。TikTok \to w \to Twitter という媒介経路を立て、総効果＝直接＋媒介の分解で伝播仮説を検証できます。
- これは CRD の見方（個の誤差 + 集団情報）とも整合し、「合意としての w」が説明力を持つ理由を与えます。 ￼

⸻

## 7. 動かし方（要約）
- Python 3.9 系で Notebook を開き、上から順に実行。
- `langgraph`, `pydantic>=2`, `numpy`, `pandas`, `scipy`, `matplotlib` などが必要です。
- MemorySaver（LangGraph）を使うため、`config={"configurable":{"thread_id":"cpc-demo-thread"}}` を `graph.invoke` に渡します（Notebook では既に修正済み）。
- チェックポイントの都合で**巨大オブジェクト（DataFrame など）は直列化（to_dict/np.array→list）**して state に保存しています（Notebook 実装済み）。

⸻

## 8. 制限と拡張（現状のデモと今後）
- ダミーデータなので、統計的に厳密な主張はしません。
- 因果は遅れ回帰＋媒介分解の玩具版です。実運用では DAG 学習／IV／差分法／因果発見等を併用してください。
- **CPC のスケールアップ**：大規模 o\to z は表現学習（例：テキスト埋め込み）、z\to w は VQ/VB 系クラスタリングやベイズ混合（CPC の one‑hot/soft 割当の連続化）が実務的です（新しい通信の数学的理論 p.6–7 の議論）。 ￼
- **アクティブ・インファレンス**は玩具です。式(2.8)–(2.10) に忠実な期待値計算や報酬設計 C_k を拡充すれば、実験計画に直接使えます（CPC‑MS p.8–10）。 ￼

⸻

## 9. 付録：最小の式で CPC をもう一度

### 9.1 生成と推論（たった 2 本）
- **生成**：\ p(w,z,o)=p(w)\prod_k p(z_k\mid w)p(o_k\mid z_k)
- **推論**：\ q(w,z\mid o)=q(w\mid z)\prod_k q(z_k\mid o_k)
（CPC‑MS 図3 p.5・式(2.1)–(2.2)） ￼

### 9.2 目的関数（自由エネルギーの分解）

F\ =\ \underbrace{\mathbb{E}[\ln q(w\mid z)]}{\text{collective}}
-\sum_k \underbrace{\mathbb{E}[\ln p(o_k\mid z_k)]}{\text{prediction}}
-\sum_k \underbrace{\mathbb{E}\!\left[\ln \tfrac{p(z_k\mid w)}{q(z_k\mid o_k)}\right]}_{\text{individual}}
（CPC‑MS 式(2.7), p.8–9） ￼

### 9.3 コードブックの最適事後

\[
q^\*(w\mid z)\ \propto\ p(w)\,\{p(z\mid w)\}^{1/\lambda}\ \ \Rightarrow\ \ \lambda\downarrow 0 \ \text{で one‑hot}
\]
（新しい通信の数学的理論 §7, p.5–7） ￼

### 9.4 集合 RD（CRD）境界

H(z,o)-D(z,o\mid w)\le I(w;z)\le C(w;z)
（同 §5–6, p.4–5） ￼

### 9.5 分散ベイズ（MHNG）の受理率

\alpha=\min\!\Bigl(1,\ \frac{p(z_{\text{listener}}\mid w’)}{p(z_{\text{listener}}\mid w)}\Bigr)
（CPC‑MS 図2 p.4・付録B p.20） ￼

⸻

## 10. 参考（本文で触れた図）
- CPC‑MS 図3（p.5）：PGM（w\to z_k\to o_k の生成と、分散推論の分解）。 ￼
- CPC‑MS 図2（p.4）：MHNG と科学活動の対応図（分散ベイズ）。 ￼
- 新しい通信の数学的理論 p.4–5：CRD 不等式の図示、p.5–7：コードブック創発（ソフト/ハード割当の幾何直観）。 ￼

⸻

**謝辞（理論面の出典）**
- 本 README の式と概念は、CPC の新しい定式化（Collective Rate–Distortion とコードブック創発）および CPC‑MS（科学の生成モデル）に拠っています。詳細は上記 PDF の該当ページを参照してください。 ￼  ￼

⸻
