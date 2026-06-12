# LLNCS Article Guide (Springer LNCS, class v2.4)

Cheat-sheet for writing the article with `llncs.cls`. Source: `llncsdoc.pdf`.

## Golden rules
- **Do not modify `llncs.cls`.** Use standard `article` commands for the body.
- **Do not use layout commands** (`\textheight`, `\vspace` except figure/table spacing, `\headsep`, etc.). The class sets layout. Text area is fixed at 12.2 cm × 19.3 cm.
- Equations auto-numbered sequentially, arabic, right side.

## Preamble / document start
```latex
\documentclass{llncs}        % single-paper: no [runningheads] needed
\begin{document}
\title{Your Title}
\titlerunning{Short Title}   % only if title too long for running head
\author{First Last\inst{1} \and Second Author\inst{2}}
\authorrunning{Last et al.}  % optional
\institute{Inst One, City, Country\\ \email{a@x.com}
\and Inst Two, City, Country}
\maketitle                   % REQUIRED — without it the heading produces no text
\begin{abstract}
Summary text.
\end{abstract}
...
\end{document}
```
- `\title` mandatory, no end punctuation, capitalize per heading rules. Split long title with `\\`.
- `\inst{n}` ties author to nth `\institute` entry (separated by `\and`, auto-numbered in order).
- `\email{}` and `\footnote`/`\thanks{}` go inside heading commands. Multiple footnotes on one item: separate with `\fnmsep`.
- `\subtitle{}` optional.
- Running heads + ToC are the volume editor's job, not the single-paper author's.

## Class options
| Option | Effect |
|---|---|
| `[runningheads]` | enable running heads (editors) |
| `[envcountsame]` | all theorem-like envs share one counter |
| `[envcountreset]` | reset theorem counters each section |
| `[envcountsect]` | number theorems by section (e.g. 7.2.1) |
| `[oribibl]` | keep original LaTeX bibliography + `\cite` (for BibTeX tools) |
| `[citeauthoryear]` | author–year citation system |
| `[orivec]` | revive original arrow vector symbol |

## Headings
- `\section{}` and `\subsection{}` — **no** end punctuation.
- `\subsubsection{}` and `\paragraph{}` — **do** punctuate at end.
- Capitalize all words in headings/titles EXCEPT conjunctions, prepositions (on, of, by, and, or, but, from, with, without, under) and articles (the, a, an) — unless first word.

## Theorem-like environments (built in)
Bold label + italic body: `theorem`, `lemma`, `corollary`, `proposition`.
Italic label + roman body: `definition`, `example`, `exercise`, `note`, `problem`, `question`, `remark`, `solution`.
`proof` — unnumbered, ends with `\qed` square.
Define new: `\spnewtheorem{name}[shares-counter]{Caption}{cap_font}{body_font}` (`\spnewtheorem*` = unnumbered).

## Math / formulas
- Math mode = italic. Non-math words inside formulas → `\mbox{...}`.
- Roman for: subscript/superscript labels (`T_\mathrm{eff}`), physical units (`\mathrm{Hz}`), abbreviations, chemical symbols. Functions `\log \sin \exp \max \sup` auto-roman.
- Foreign phrases (et al., a priori, in situ) NOT italic.
- Use `\left( \right)` for auto-sized delimiters. Punctuate displayed equations; `\enspace` before end punctuation.
- New paragraph after a displayed equation: leave blank line (gives indent). Continuing same paragraph: `\noindent` or no blank line.

## Figures
- Place figure environment AFTER (not inside) the paragraph that first mentions it. Auto-numbered.
- Caption BELOW figure. Prefer EPS via `\includegraphics`.
```latex
\begin{figure}
\centering
\includegraphics{...}
\caption[Short caption]{Full caption.}  % type [ ] / short form per project convention
\end{figure}
```

## Tables
- Caption ABOVE table. Auto-numbered. Treat caption like figure legend (no capitalization of normal words).
```latex
\begin{table}
\caption{Critical $N$ values}
\begin{tabular}{llllll}
\hline\noalign{\smallskip}
header & ... \\
\noalign{\smallskip}\hline\noalign{\smallskip}
data & ... \\
\noalign{\smallskip}\hline
\end{tabular}
\end{table}
```
- Need an empty line before continuing text after a table.

## Capitalization rules (text body)
- ALWAYS capitalize: headings; abbreviations Fig., Table, Sect., Chap., Theorem, Definition **when followed by a number** (Fig. 3, Table 1).
- DON'T capitalize: figure/table/equation/theorem when used WITHOUT a number; figure legends & table captions (except names/abbreviations).

## Abbreviations
- Abbreviate Chap., Sect., Fig. in running text UNLESS sentence-initial ("Figure 9 reveals...").
- Equations referenced by number in parens: "(14)". Sentence-initial: "Equation (14)".
- Define acronym at first use: "Strong Optimization (SOPT) Problem".

## Text fine-tuning
| Code | Use |
|---|---|
| `\,` | thin space (numbers, units: `21\,$^{\circ}$C`, `20\,000`) |
| `--` | en dash (ranges: `1950--1985`) |
| ` -- ` | en dash with spaces (parenthetical) |
| `-` | hyphen |
| `$-$` | minus sign in text |

## Emphasis
- Italic: `\emph{}` (preferred) or `{\itshape ...}`. Bold: `{\bfseries ...}`.
- Vectors are bold by default (LNCS convention): `$\vec{a}$` → bold **a**. Math mode only.

## Lists
Standard `enumerate`/`itemize`; nesting supported (1. → (a) → ...).

## Footnotes
`\footnote{Text}` — auto-numbered.

## References / bibliography
- One citation system only. **Number-only is preferred.**
- BibTeX style: `splncs.bst` → `\bibliographystyle{splncs}`. For a custom BibTeX style add `[oribibl]` option.
- Number-only: omit optional arg of `\bibitem`. Citations collapse to ranges, e.g. `\cite{a,b,c}` → [1-3].
```latex
\begin{thebibliography}{1}
\bibitem{clar:eke}
Clarke, F., Ekeland, I.: Nonlinear oscillations and boundary-value problems for
Hamiltonian systems. Arch. Rat. Mech. Anal. \textbf{78} (1982) 315--333
\end{thebibliography}
```
- Author–year: add `[citeauthoryear]`, write author name in text, `\cite` supplies year, `\bibitem[year]{key}`.

## Program code
Use `verbatim` environment / package.
