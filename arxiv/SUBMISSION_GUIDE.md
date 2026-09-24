# arXiv 提交说明 / Submission guide

## 文件
- `arxiv_source.tar.gz`：上传到 arXiv 的源码包（main.tex、main.bbl、tab_e3.tex、figs/*.pdf）。已在空目录中仅用 pdflatex（不跑 bibtex）编译通过，16 页，无错误、无未定义引用。
- `jev-hamiltonian.pdf`：由同一源码编译出的 PDF，用于核对 arXiv 生成的预览。
- `abstract.txt`：纯文本摘要（1,920 字符以内，已去掉 LaTeX 宏；比论文摘要略短），直接粘贴到摘要栏。
- `repo_snapshot.zip`：GitHub 仓库当前提交的完整快照（代码、缓存数据、审稿记录），与论文中的代码地址对应。

## 表单填写
- **Title:** Is there a Hamiltonian inside a decision model? Probing the energy landscape of Jev for quantum chemistry
- **Authors:** Zhaohe Dong
- **Abstract:** 粘贴 `abstract.txt`
- **Primary category:** cs.LG (Machine Learning)
- **Cross-lists:** physics.chem-ph, quant-ph
- **Comments:** 16 pages, 6 figures, 1 table. Code and cached data: https://github.com/dongzhaohe321418-lab/jev-hamiltonian
- **License:** 建议 CC BY 4.0（最开放）；若想保留更多权利可选 arXiv 默认的 non-exclusive license。
- **Journal-ref / DOI:** 留空。

## 提交前你需要确认
1. **仓库已公开**（2026-09-24），论文中的代码地址可以访问。
2. **背书 (endorsement)。** 首次在 cs.LG 提交可能需要背书。若系统提示，需找一位已在 cs.LG 发表过的作者用你收到的背书码为你背书。
3. **AI 使用声明。** 论文已写明：代码、实验、文献整理和初稿由 Claude 完成，审稿为 AI 模拟、未经人工同行评审；作者对内容负责。arXiv 要求作者本人对全部内容负责，提交前请通读一遍。
4. **API 密钥。** 你曾在对话中贴出 Jev 密钥，建议提交前在 TypeSafe 控制台轮换。仓库里没有密钥。
5. **邮箱。** 论文首页写了 dongzhaohe321418@gmail.com，会公开显示；如不希望公开可告诉我删掉。

## 步骤
1. 登录 https://arxiv.org/submit → Start new submission。
2. 选 license、primary category cs.LG。
3. 上传 `arxiv_source.tar.gz`，等待自动编译，核对预览与 `jev-hamiltonian.pdf` 一致。
4. 填元数据（见上），加 cross-list physics.chem-ph、quant-ph。
5. 预览无误后提交。通常下一个工作日公布。
