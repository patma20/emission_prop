%%%% Classe LaTex - ICAS - Last update 01/03/2020 - Olivia Fischer 
\NeedsTeXFormat{LaTeX2e}
\RequirePackage{graphicx}
\RequirePackage[utf8]{inputenc}
\RequirePackage{setspace}
\RequirePackage{ifthen}
\newif\if@submission
\@submissionfalse
\DeclareOption{submission}{\global\@submissiontrue}

\ProcessOptions
\LoadClass[11pt,a4paper]{article}


\usepackage{indentfirst}
\usepackage{setspace}
\usepackage{authblk}
\renewcommand\Authsep{, }
\renewcommand\Authand{ \& }
\renewcommand\Authands{ \& }

\usepackage{fancyhdr}
\usepackage{geometry}
\usepackage{lastpage}
\usepackage{lineno}
\usepackage[]{hyperref}
\hypersetup{}
\usepackage[english]{babel}
\usepackage{epstopdf}
\usepackage{graphicx}
\usepackage{lipsum}
\usepackage{enumerate}
\usepackage{wrapfig,framed}
\usepackage[]{caption}
\captionsetup[table]{font=normalsize,justification=justified,singlelinecheck=off,labelsep=endash}
\captionsetup[figure]{font=normalsize,justification=centering,singlelinecheck=off,labelsep=endash}
\usepackage[]{subcaption}

\usepackage{amsfonts}
\usepackage{amssymb}
\usepackage{amsmath}


% \usepackage{helvet}
% \DeclareCaptionFont{\sffamily}

%  font
\usepackage{helvet}
\usepackage{times}
\usepackage{mathptmx}
\usepackage[T1]{fontenc}
\usepackage{xcolor}
\definecolor{azulcen}{rgb}{0.1211,0.2813,0.4609}
\newcommand{\fonteppp}{\scriptsize}
\newcommand{\fontepp}{\footnotesize}
\newcommand{\fontep}{\small}
\newcommand{\fonten}{\normalsize}
\newcommand{\fonteg}{\large}
\newcommand{\fontegg}{\Large}
\newcommand{\fonteggg}{\LARGE}

% margins
\setlength{\paperheight}{297mm}
\setlength{\paperwidth}{210mm}
\setlength{\voffset}{-2cm}
\setlength{\topmargin}{0cm}
\setlength{\headheight}{1cm}
\setlength{\headsep}{0.5cm}
\setlength{\footskip}{0cm}
\setlength{\textheight}{25.7cm}
\setlength{\oddsidemargin}{-0.5cm}
\setlength{\hoffset}{0cm}
\setlength{\marginparsep}{0mm}
\setlength{\marginparwidth}{0mm}    
\setlength{\marginparpush}{0mm}
\setlength{\marginparwidth}{0mm}    
\setlength{\textwidth}{17cm}
\setlength{\parindent}{0cm}

    \def\ps@ICAShead{%
        
      \def\@oddfoot{%
      \parbox{\textwidth}{\begin{center}\sffamily\fontep \thepage\end{center}}}%
      \let\@evenfoot\@empty

      \def\@oddhead{%
      %\parbox{\textwidth}{\begin{flushright}\sffamily\fontep \@title\end{flushright}}}%
      
 %%%%% THIS IS WHERE YOU MODIFY THE TITLE THAT APPEARS AT THE TOP OF EACH PAGE %%%%%%
 %%%%% The title needs to fit in a single line - if too long please use a "running title"
 %%%%% i.e. a simplification of the title, or a shorter version of the title, such that it fits in one single line
      \parbox{\textwidth}{\begin{flushright}\sffamily\bfseries\fontepp{Low-Emission Propulsion Analysis and Optimization of a High Bypass Turbofan}\end{flushright}}}%
      

      %\parbox{\textwidth}{\begin{flushright}\sffamily\fontep \thepage\end{flushright}}}%
      %\let\@oddhead\@title

     % \def\@evenhead{%
     % \parbox{\textwidth}{\begin{flushright}\sffamily\fontep \@author\end{flushright}}}%
     % \parbox{\textwidth}{\begin{center}\sffamily\fontep \thepage\end{center}}}%
      %\let\@evenhead\@author

}


    \def\ps@ICASheadum{%
        \let\@oddhead\@empty
        \let\@oddfoot\@empty
        \let\@evenhead\@empty
        \let\@evenfoot\@empty
}



\renewcommand{\section}{
   \@startsection{section}{1}{\z@}%
   {-11pt \@plus -1ex \@minus -.2ex}% spacing before
   {.3ex \@plus .2ex}% spacing after
   {\reset@font\fonteg\sffamily\bfseries\color{azulcen}}}
\renewcommand{\thesection}{\arabic{section}.\hspace{-1ex}}

\renewcommand{\subsection}{
   \@startsection{subsection}{1}{\z@}%
   {-11pt \@plus -1ex \@minus -.2ex}% spacing before
   {.3ex \@plus .2ex}% spacing after
   {\reset@font\fonteg\sffamily\color{azulcen}}}
\renewcommand{\thesubsection}{\hspace{\parindent}\arabic{section}.\arabic{subsection}\hspace{-1ex}}

\renewcommand{\subsubsection}{
   \@startsection{subsubsection}{1}{\z@}%
   {-11pt \@plus -1ex \@minus -.2ex}% spacing before
   {.3ex \@plus .2ex}% spacing after
   {\reset@font\fonteg\sffamily\itshape\color{azulcen}}}
\renewcommand{\thesubsubsection}{\hspace{\parindent}\arabic{section}.\arabic{subsection}.\arabic{subsubsection}\hspace{-1ex}}


% Define title, authors, affiliations, abstract, keywords
\newcommand{\TitlePaper}{\title}
\newcommand{\AuthorPaper}{\author}
\newcommand{\abstractEnglish}{\newcommand{\abst}}
\newcommand{\keywords}{\newcommand{\key}}

\newcommand{\ICAS@pagum}{%

    \begin{flushright}
  \includegraphics[width=1\textwidth]{../figures/banner2022.jpg}
    \end{flushright}



    \begin{center}
    {\noindent{\sffamily\bfseries \color{azulcen}\centering\fontegg\MakeUppercase\@title} \par}%
    \vspace{10pt}
    \renewcommand\Authfont{}
    \renewcommand\Affilfont{\fontepp}
    
    \if@submission

    \else

    \noindent\sffamily\fonten\@author
    \fi
    
%     \renewcommand\Authfont{}
%     \renewcommand\Affilfont{\sffamily\fontep}
%     \sffamily\fonten\@author

    \end{center}
    \vspace{\parskip}
    
    \begin{center}{\noindent\sffamily\bfseries\color{azulcen}\fonten Abstract}\end{center}
    \par {\sffamily\fontep\abst}
    \vspace{5pt}
    \par {\sffamily\fontep\textbf{Keywords:~} \key\\}
    
    }


\newenvironment{cover}{}{\initiate}

\newcommand\initiate{
    \thispagestyle{ICASheadum}
    \if@twocolumn
	  \ifnum \col@number=\@ne
	    \ICAS@pagum
	  \else
	    \twocolumn[\ICAS@pagum]%
	  \fi
    \else
	  \global\@topnum\z@   % Prevents figures from going at top of page.
	  \ICAS@pagum
    \fi
    
}



% % % % Biblio
\newcommand\biblio[1]{
   \bibliography{#1}
}

\renewenvironment{thebibliography}[1]
     {\section*{\refname
        \@mkboth{\MakeUppercase\refname}{\MakeUppercase\refname}}%
      \list{\small\@biblabel{\@arabic\c@enumiv}}%
           {\settowidth\labelwidth{\@biblabel{#1}}%
            \leftmargin\labelwidth
            \advance\leftmargin\labelsep
            \usecounter{enumiv}%
            \let\p@enumiv\@empty
            \renewcommand\theenumiv{\@arabic\c@enumiv}
      \@bibsetup}%
      \small\renewcommand\newblock{\hskip .11em \@plus.33em \@minus.07em}%
      \sloppy\clubpenalty4000\widowpenalty4000%
      \sfcode`\.=\@m}
     {\def\@noitemerr
       {\@latex@warning{Empty `thebibliography' environment}}%
      \endlist}

%% setup 8/10 type
\def\@bibsetup{%\itemindent=0pt
  \itemsep=2pt \parsep=0pt \parskip 8pt
  \small}

\def\sameauthor{\leavevmode\vrule height 2pt depth -1.6pt width 23pt}

    % \renewenvironment{thebibliography}[1]
    % {\section*{\bibname}%
    % \list{\@biblabel{\@arabic\c@enumiv}}%
    % {\settowidth\labelwidth{\@biblabel{#1}}%
    % \leftmargin\labelwidth
    % \advance\leftmargin\labelsep
    % \@openbib@code
    % \usecounter{enumiv}%
    % \let\p@enumiv\@empty
    % \renewcommand\theenumiv{\@arabic\c@enumiv}}%
    % \singlespacing
    % \sloppy
    % \clubpenalty4000
    % \@clubpenalty \clubpenalty
    % \widowpenalty4000%
    % \sfcode`\.\@m}
    % {\def\@noitemerr
    % {\@latex@warning{Empty `thebibliography' environment}}%
    % \endlist}

\newcommand\body{
    \pagestyle{ICAShead}
    \initiate
    \sffamily\fonten
    \renewcommand{\rmdefault}{phv} % Arial
    \renewcommand{\sfdefault}{phv} %Arial
 } 


\endinput
