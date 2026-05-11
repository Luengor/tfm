# LaTeX Elements Guide

Reference for elements used in `informe.tex`.

---

## Sections and labels

```latex
\section{Introduction}
\label{sec:intro}

\subsection{Sub Section 4.1}
\label{sec:41sub}

\subsubsection{Sub Sub Section 5.1.1}
\label{sec:511subsub}
```

Paragraphs (depth 4) are enabled via `\setcounter{secnumdepth}{4}` in the preamble.

Cross-reference a label anywhere with `\ref{sec:intro}`.

---

## Lists

### Bullet list (itemize)

```latex
\textbf{Items:}

\begin{itemize}
  \item First item.
  \item Second item.
  \item Third item.
\end{itemize}
```

### Numbered list (enumerate) with nesting

```latex
\textbf{Enumerations:}

\begin{enumerate}
  \item 1.1 Element.
  \item 1.2 Element.
  \begin{enumerate}
    \item 2.1 Element.
    \item 2.2 Element.
  \end{enumerate}
  \item 1.3 Element.
\end{enumerate}
```

---

## Tables

Uses the `longtable` package (handles tables that span multiple pages). No surrounding `table` float is needed.

```latex
\begin{longtable}{|p{7cm}|p{7cm}|} \hline
  \textbf{Column 1} & \textbf{Column 2} \\ \hline
  Cell 1.1 & Cell 1.2 \\ \hline
  Cell 2.1 & Cell 2.2\footnotemark[1] \textbf{With Footnote} \\ \hline
  Cell 3.1 & Cell 3.2 \\ \hline
  \caption{Two-column table}
  \label{tab:twoColumns}
\end{longtable}

\footnotetext[1]{This is the footnote text.}
```

**Column spec quick reference:**

| Spec | Meaning |
|------|---------|
| `p{7cm}` | Fixed-width column, text wraps |
| `\|` | Vertical border |
| `\hline` | Horizontal border |
| `\\` | End of row |
| `&` | Column separator |

The `\caption` and `\label` go **inside** the `longtable` environment, before `\end{longtable}`.

Footnotes inside tables use `\footnotemark[N]` in the cell and `\footnotetext[N]{...}` **after** `\end{longtable}`.

---

## Verbatim / code listings

Uses the `fancyvrb` package (`\usepackage{fancyvrb}`).

### Plain verbatim block (logs, terminal output)

```latex
\begin{Verbatim}[fontsize=\scriptsize]
Wed Jul  5 06:21:26 2006 OpenVPN 2.0.7 i686-pc-linux [SSL] built on Jun 6 2006
Wed Jul  5 06:21:26 2006 Initialization Sequence Completed
\end{Verbatim}
```

### Source code block

Same environment — `Verbatim` preserves all whitespace and indentation exactly:

```latex
\begin{Verbatim}[fontsize=\scriptsize]
do {
    retval = recv(asd, (char *)recvbuf, BUF_SIZE, MSG_TIMEOUT, 20000L, &error);
    if (retval == API_ERROR) {
        break;
    }
} while (retval > 0);
\end{Verbatim}
```

Available `fontsize` options (smallest → largest): `\tiny`, `\scriptsize`, `\footnotesize`, `\small`, `\normalsize`.

> The `listings` package is also loaded (`\usepackage{listings}`) with Matlab as the default language. Use `lstlisting` for syntax-highlighted code:
> ```latex
> \begin{lstlisting}
> x = sin(pi/4);
> \end{lstlisting}
> ```

---

## Figures

Requires `\usepackage{wrapfig}` (for text-wrapped figures) and `\usepackage{epsfig}`.

### Centered figure (no text wrap)

```latex
\begin{figure}[ht]
  \centering
  \includegraphics[width=5.5cm]{figures/my_image.jpg}
  \caption{Centered figure without text}
  \label{Fig:centered}
\end{figure}
```

### Figure floated left with text wrap

```latex
\begin{wrapfigure}[11]{l}[1pt]{8.5cm}
  \includegraphics[width=8.5cm, height=3.5cm]{figures/my_image.jpg}
  \caption{Image on the left}
  \label{Fig:left}
\end{wrapfigure}

Paragraph text that flows around the image...
```

`[11]` — number of text lines to wrap; `{l}` — side (`l`/`r`); `[1pt]` — overhang; `{8.5cm}` — figure width.

### Figure floated right with text wrap

```latex
\begin{wrapfigure}[11]{r}[1pt]{8.5cm}
  \includegraphics[width=8.5cm, height=3.5cm]{figures/my_image.jpg}
  \caption{Image on the right}
  \label{Fig:right}
\end{wrapfigure}

Paragraph text that flows around the image...
```

### Two figures side by side

```latex
\begin{figure}
  \begin{minipage}{.5\linewidth}
    \centering
    \includegraphics[width=7cm]{figures/image1.jpg}
    \caption{Image 1}
    \label{Fig:side1}
  \end{minipage}%
  \begin{minipage}{.5\linewidth}
    \centering
    \includegraphics[width=4cm]{figures/image2.jpg}
    \caption{Image 2}
    \label{Fig:side2}
  \end{minipage}
\end{figure}
```

The `%` after `\end{minipage}` suppresses the inter-minipage space.

---

## Text formatting

| Effect | Command |
|--------|---------|
| **Bold** | `\textbf{text}` |
| *Italic / emphasis* | `\emph{text}` |
| Strikethrough / underline | `\usepackage{ulem}` then `\sout{}`, `\uline{}` |

---

## Appendices

```latex
\appendix

\renewcommand{\thesection}{\Alph{section}}
\renewcommand{\thesubsection}{\thesection.\arabic{subsection}}

\newpage
\section{Appendix A - Images}
\label{Apx:AppendixA}
```

After `\appendix`, sections are lettered (A, B, C…) automatically.

---

## Bibliography

Add citations in text:

```latex
Some books about LaTeX: \cite{kreher05pseudocode}, \cite{ohl97emp}.
```

At the end of the document:

```latex
\newpage
\bibliographystyle{plain}
\bibliography{bib}
```

Where `bib` refers to `bib.bib` in the same directory.

---

## Page breaks and spacing

```latex
\newpage          % Force a new page

\setlength{\parskip}{2.5mm}   % Paragraph spacing (set in preamble or mid-document)
```
