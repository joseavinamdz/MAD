# MAD
Artificial Intellingence

# Artificial Intelligence: Machine Learning

## _Logistic Regression_


### Sigmoid Function 
... Definition
<p align="center"
><img src="sigmoid_function.png" />
![Paper written in LaTeX]
</p>

***

#### Lower and Upper limits
If we use the limits:

<img src="https://tex.s2cms.ru/svg/%0A%5Clarge%7B%0A%5Cbegin%7Barray%7D%7Bl%7D%5Clim%20_%7Bx%20%5Crightarrow%20%5Cinfty%7D%20%5Cmathrm%7Be%7D%5E%7B-x%7D%20%5Crightarrow%200%20%5C%5C%20%5C%5C%20%5Clim%20_%7Bx%20%5Crightarrow-%5Cinfty%7D%20%5Cmathrm%7Be%7D%5E%7B-x%7D%20%5Crightarrow%20%5Cinfty%5Cend%7Barray%7D%7D%0A" alt="
\large{
\begin{array}{l}\lim _{x \rightarrow \infty} \mathrm{e}^{-x} \rightarrow 0 \\ \\ \lim _{x \rightarrow-\infty} \mathrm{e}^{-x} \rightarrow \infty\end{array}}
" />

this leads to:

<img src="https://tex.s2cms.ru/svg/%0A%5CLarge%7B%0A%5Cbegin%7Barray%7D%7Bl%7D%5Clim%20_%7Bx%20%5Crightarrow%20%5Cinfty%7D%20%5Cfrac%7B1%7D%7B1%2Be%5E%7B-x%7D%7D%3D%5Cfrac%7B1%7D%7B1%2B%5Clim%20_%7Bx%20%5Crightarrow%20%5Cinfty%7D%20e%5E%7B-x%7D%7D%3D%5Cfrac%7B1%7D%7B1%2B0%7D%3D1%20%5C%5C%20%5C%5C%0A%5Clim%20_%7Bx%20%5Crightarrow-%5Cinfty%7D%20%5Cfrac%7B1%7D%7B1%2Be%5E%7B-x%7D%7D%3D%5Cfrac%7B1%7D%7B1%2B%5Clim%20_%7Bx%20%5Crightarrow-%5Cinfty%7D%20e%5E%7B-x%7D%7D%3D%5Cfrac%7B1%7D%7B1%2B%5Clim%20_%7Bx%20%5Crightarrow%20%5Cinfty%7D%20e%5E%7Bx%7D%7D%3D0%5Cend%7Barray%7D%7D%0A" alt="
\Large{
\begin{array}{l}\lim _{x \rightarrow \infty} \frac{1}{1+e^{-x}}=\frac{1}{1+\lim _{x \rightarrow \infty} e^{-x}}=\frac{1}{1+0}=1 \\ \\
\lim _{x \rightarrow-\infty} \frac{1}{1+e^{-x}}=\frac{1}{1+\lim _{x \rightarrow-\infty} e^{-x}}=\frac{1}{1+\lim _{x \rightarrow \infty} e^{x}}=0\end{array}}
" />

***

#### Derivation of Sigmoid Function

<img src="https://tex.s2cms.ru/svg/%5CLarge%7B%0A%5Cbegin%7Barray%7D%7Bl%7D%0A%5Csigma(x)%3D%5Cfrac%7B1%7D%7B1%2Be%5E%7B-x%7D%7D%20%5C%20%5C%20%5C%20%5C%20%5C%20%5C%0A%5Cend%7Barray%7D%7D%0A" alt="\Large{
\begin{array}{l}
\sigma(x)=\frac{1}{1+e^{-x}} \ \ \ \ \ \
\end{array}}
" />

Right, let’s start deriving the sigmoid function.
So, we want the value of:

<img src="https://tex.s2cms.ru/svg/%5CLarge%7B%0A%5Cbegin%7Barray%7D%7Bl%7D%0A%5Csigma%5E%7B%5Cprime%7D(x)%3D%5Cfrac%7Bd%7D%7Bd%20x%7D%20%5Csigma(x)%3D%5Cfrac%7Bd%7D%7Bd%20x%7D%20%5Cfrac%7B1%7D%7B1%2Be%5E%7B-x%7D%7D%0A%5Cend%7Barray%7D%7D%0A" alt="\Large{
\begin{array}{l}
\sigma^{\prime}(x)=\frac{d}{d x} \sigma(x)=\frac{d}{d x} \frac{1}{1+e^{-x}}
\end{array}}
" />

Next, we will apply the reciprocal rule

<img src="https://tex.s2cms.ru/svg/%0A%5CLarge%7B%0A%5Cleft%5B%5Cfrac%7B1%7D%7Bu(x)%7D%5Cright%5D%5E%7B%5Cprime%7D%3D%5Cleft%5Bu(x)%5E%7B-1%7D%5Cright%5D%5E%7B%5Cprime%7D%3D-%5Cleft%5B%5Cfrac%7Bu%5E%7B%5Cprime%7D(x)%7D%7Bu(x)%5E%7B2%7D%7D%5Cright%5D%3D-u(x)%5E%7B-2%7D%20%5Ccdot%20u%5E%7B%5Cprime%7D(x)%7D%0A" alt="
\Large{
\left[\frac{1}{u(x)}\right]^{\prime}=\left[u(x)^{-1}\right]^{\prime}=-\left[\frac{u^{\prime}(x)}{u(x)^{2}}\right]=-u(x)^{-2} \cdot u^{\prime}(x)}
" />

Applying the reciprocal rule, takes us to the next step

<img src="https://tex.s2cms.ru/svg/%0A%5CLarge%7B%0A%3D%5Cfrac%7Bd%7D%7Bd%20x%7D%5Cleft(1%2Be%5E%7B-x%7D%5Cright)%5E%7B-1%7D%3D-%5Cleft(1%2Be%5E%7B-x%7D%5Cright)%5E%7B-2%7D%20%5Ccdot%20%5Cfrac%7Bd%7D%7Bd%20x%7D%5Cleft(1%2Be%5E%7B-x%7D%5Cright)%7D%0A" alt="
\Large{
=\frac{d}{d x}\left(1+e^{-x}\right)^{-1}=-\left(1+e^{-x}\right)^{-2} \cdot \frac{d}{d x}\left(1+e^{-x}\right)}
" />

Next, we need to apply the rule of linearity, which simply says

<img src="https://tex.s2cms.ru/svg/%0A%5CLarge%7B%5Ba%20.%20u(x)%2Bb%20.%20v(x)%5D%5E%7B%5Cprime%7D%3Da%20%5Ccdot%20u%5E%7B%5Cprime%7D(x)%2Bb%20.%20v%5E%7B%5Cprime%7D(x)%7D%0A" alt="
\Large{[a . u(x)+b . v(x)]^{\prime}=a \cdot u^{\prime}(x)+b . v^{\prime}(x)}
" />

Applying the rule of linearity, we get

<img src="https://tex.s2cms.ru/svg/%5CLarge%7B%0A%3D-%5Cleft(1%2Be%5E%7B-x%7D%5Cright)%5E%7B-2%7D%20%5Ccdot%20%5Cfrac%7Bd%7D%7Bd%20x%7D%5Cleft(1%2Be%5E%7B-x%7D%5Cright)%3D-%5Cleft(1%2Be%5E%7B-x%7D%5Cright)%5E%7B-2%7D%20%5Ccdot%5Cleft(%5Cfrac%7Bd%7D%7Bd%20x%7D%5B1%5D%2B%5Cfrac%7Bd%7D%7Bd%20x%7D%5Cleft%5Be%5E%7B-x%7D%5Cright%5D%5Cright)%7D%0A" alt="\Large{
=-\left(1+e^{-x}\right)^{-2} \cdot \frac{d}{d x}\left(1+e^{-x}\right)=-\left(1+e^{-x}\right)^{-2} \cdot\left(\frac{d}{d x}[1]+\frac{d}{d x}\left[e^{-x}\right]\right)}
" />

<img src="https://tex.s2cms.ru/svg/%5Cldots" alt="\ldots" /> now let’s derive each of them one by one.
Now, derivative of a constant is 0, so we can write the next step as

<img src="https://tex.s2cms.ru/svg/%5CLarge%7B%0A%3D-%5Cleft(1%2Be%5E%7B-x%7D%5Cright)%5E%7B-2%7D%20%5Ccdot%5Cleft(%5Cfrac%7Bd%7D%7Bd%20x%7D%5B1%5D%2B%5Cfrac%7Bd%7D%7Bd%20x%7D%5Cleft%5Be%5E%7B-x%7D%5Cright%5D%5Cright)%3D-%5Cleft(1%2Be%5E%7B-x%7D%5Cright)%5E%7B-2%7D%20%5Ccdot%5Cleft(0%2B%5Cfrac%7Bd%7D%7Bd%20x%7D%5Cleft%5Be%5E%7B-x%7D%5Cright%5D%5Cright)%7D%0A" alt="\Large{
=-\left(1+e^{-x}\right)^{-2} \cdot\left(\frac{d}{d x}[1]+\frac{d}{d x}\left[e^{-x}\right]\right)=-\left(1+e^{-x}\right)^{-2} \cdot\left(0+\frac{d}{d x}\left[e^{-x}\right]\right)}
" />

And adding 0 to something doesn’t effects so we will be removing the 0 in the next step and moving with the next derivation for which we will require the exponential rule, which simply says

<img src="https://tex.s2cms.ru/svg/%0A%5CLarge%7B%5Cleft%5Be%5E%7Bu(x)%7D%5Cright%5D%5E%7B%5Cprime%7D%3De%5E%7Bu(x)%7D%20%5Ccdot%20u%5E%7B%5Cprime%7D(x)%7D%0A" alt="
\Large{\left[e^{u(x)}\right]^{\prime}=e^{u(x)} \cdot u^{\prime}(x)}
" />

Applying the exponential rule we get,

<img src="https://tex.s2cms.ru/svg/%5CLarge%7B%0A%3D-%5Cleft(1%2Be%5E%7B-x%7D%5Cright)%5E%7B-2%7D%20%5Ccdot%5Cleft(%5Cfrac%7Bd%7D%7Bd%20x%7D%5Cleft%5Be%5E%7B-x%7D%5Cright%5D%5Cright)%3D-%5Cleft(1%2Be%5E%7B-x%7D%5Cright)%5E%7B-2%7D%20%5Ccdot%5Cleft(e%5E%7B-x%7D%20%5Ccdot%20%5Cfrac%7Bd%7D%7Bd%20x%7D%5B-x%5D%5Cright)%7D%0A" alt="\Large{
=-\left(1+e^{-x}\right)^{-2} \cdot\left(\frac{d}{d x}\left[e^{-x}\right]\right)=-\left(1+e^{-x}\right)^{-2} \cdot\left(e^{-x} \cdot \frac{d}{d x}[-x]\right)}
" />

Again, to better understand you can simply replace <img src="https://tex.s2cms.ru/svg/e%5E%7Bu(x)%7D" alt="e^{u(x)}" /> in the exponential rule with <img src="https://tex.s2cms.ru/svg/%20e%5E%7B(-x)%7D%20" alt=" e^{(-x)} " />

Next, by the rule of linearity we can write

<img src="https://tex.s2cms.ru/svg/%5CLarge%7B%0A%3D-%5Cleft(1%2Be%5E%7B-x%7D%5Cright)%5E%7B-2%7D%20%5Ccdot%5Cleft(e%5E%7B-x%7D%20%5Ccdot%20%5Cfrac%7Bd%7D%7Bd%20x%7D%5B-x%5D%5Cright)%3D-%5Cleft(1%2Be%5E%7B-x%7D%5Cright)%5E%7B-2%7D%20%5Ccdot%5Cleft(e%5E%7B-x%7D%20%5Ccdot-%5Cfrac%7Bd%7D%7Bd%20x%7D%5Bx%5D%5Cright)%7D%0A" alt="\Large{
=-\left(1+e^{-x}\right)^{-2} \cdot\left(e^{-x} \cdot \frac{d}{d x}[-x]\right)=-\left(1+e^{-x}\right)^{-2} \cdot\left(e^{-x} \cdot-\frac{d}{d x}[x]\right)}
" />

Derivative of the differentiation variable is 1, applying which we get

<img src="https://tex.s2cms.ru/svg/%0A%3D-%5Cleft(1%2Be%5E%7B-x%7D%5Cright)%5E%7B-2%7D%20%5Ccdot%5Cleft(e%5E%7B-x%7D%20%5Ccdot-%5Cfrac%7Bd%7D%7Bd%20x%7D%5Bx%5D%5Cright)%3D-%5Cleft(1%2Be%5E%7B-x%7D%5Cright)%5E%7B-2%7D%20%5Ccdot%5Cleft(e%5E%7B-x%7D%20%5Ccdot-1%5Cright)%0A" alt="
=-\left(1+e^{-x}\right)^{-2} \cdot\left(e^{-x} \cdot-\frac{d}{d x}[x]\right)=-\left(1+e^{-x}\right)^{-2} \cdot\left(e^{-x} \cdot-1\right)
" />

Now, we can simply open the second pair of parenthesis and applying the basic rule <img src="https://tex.s2cms.ru/svg/-1%20*%20-1%20%3D%20%2B%20%5C%201" alt="-1 * -1 = + \ 1" /> we get

<img src="https://tex.s2cms.ru/svg/%0A%3D-%5Cleft(1%2Be%5E%7B-x%7D%5Cright)%5E%7B-2%7D%20%5Ccdot%5Cleft(e%5E%7B-x%7D%20%5Ccdot-1%5Cright)%3D%5Cleft(1%2Be%5E%7B-x%7D%5Cright)%5E%7B-2%7D%20%5Ccdot%20e%5E%7B-x%7D%0A" alt="
=-\left(1+e^{-x}\right)^{-2} \cdot\left(e^{-x} \cdot-1\right)=\left(1+e^{-x}\right)^{-2} \cdot e^{-x}
" />

which can be written as

<img src="https://tex.s2cms.ru/svg/%0A%3D%5Cleft(1%2Be%5E%7B-x%7D%5Cright)%5E%7B-2%7D%20%5Ccdot%20e%5E%7B-x%7D%3D%5Cfrac%7Be%5E%7B-x%7D%7D%7B%5Cleft(1%2Be%5E%7B-x%7D%5Cright)%5E%7B2%7D%7D%0A" alt="
=\left(1+e^{-x}\right)^{-2} \cdot e^{-x}=\frac{e^{-x}}{\left(1+e^{-x}\right)^{2}}
" />

Right, we are complete with the derivative. 

***

Now, we still need to simplify it a bit to get to the form used in Machine Learning. 


First, let’s rewrite it as follows

<img src="https://tex.s2cms.ru/svg/%5CLarge%7B%0A%3D%5Cfrac%7Be%5E%7B-x%7D%7D%7B%5Cleft(1%2Be%5E%7B-x%7D%5Cright)%5E%7B2%7D%7D%3D%5Cfrac%7B1%20.%20e%5E%7B-x%7D%7D%7B%5Cleft(1%2Be%5E%7B-x%7D%5Cright)%20%5Ccdot%5Cleft(1%2Be%5E%7B-x%7D%5Cright)%7D%7D%0A" alt="\Large{
=\frac{e^{-x}}{\left(1+e^{-x}\right)^{2}}=\frac{1 . e^{-x}}{\left(1+e^{-x}\right) \cdot\left(1+e^{-x}\right)}}
" />

And then rewrite it as

<img src="https://tex.s2cms.ru/svg/%0A%3D%5Cfrac%7B1%20.%20e%5E%7B-x%7D%7D%7B%5Cleft(1%2Be%5E%7B-x%7D%5Cright)%20%5Ccdot%5Cleft(1%2Be%5E%7B-x%7D%5Cright)%7D%3D%5Cfrac%7B1%7D%7B%5Cleft(1%2Be%5E%7B-x%7D%5Cright)%7D%20%5Ccdot%20%5Cfrac%7Be%5E%7B-x%7D%7D%7B%5Cleft(1%2Be%5E%7B-x%7D%5Cright)%7D%0A" alt="
=\frac{1 . e^{-x}}{\left(1+e^{-x}\right) \cdot\left(1+e^{-x}\right)}=\frac{1}{\left(1+e^{-x}\right)} \cdot \frac{e^{-x}}{\left(1+e^{-x}\right)}
" />

And since $+1 — 1 = 0$ we can do this

<img src="https://tex.s2cms.ru/svg/%0A%3D%5Cfrac%7B1%7D%7B%5Cleft(1%2Be%5E%7B-x%7D%5Cright)%7D%20%5Ccdot%20%5Cfrac%7Be%5E%7B-x%7D%7D%7B%5Cleft(1%2Be%5E%7B-x%7D%5Cright)%7D%3D%5Cfrac%7B1%7D%7B%5Cleft(1%2Be%5E%7B-x%7D%5Cright)%7D%20%5Ccdot%20%5Cfrac%7Be%5E%7B-x%7D%2B1-1%7D%7B%5Cleft(1%2Be%5E%7B-x%7D%5Cright)%7D%0A" alt="
=\frac{1}{\left(1+e^{-x}\right)} \cdot \frac{e^{-x}}{\left(1+e^{-x}\right)}=\frac{1}{\left(1+e^{-x}\right)} \cdot \frac{e^{-x}+1-1}{\left(1+e^{-x}\right)}
" />

And now let’s break the fraction and rewrite it as

<img src="https://tex.s2cms.ru/svg/%0A%3D%5Cfrac%7B1%7D%7B%5Cleft(1%2Be%5E%7B-x%7D%5Cright)%7D%20%5Ccdot%20%5Cfrac%7Be%5E%7B-x%7D%2B1-1%7D%7B%5Cleft(1%2Be%5E%7B-x%7D%5Cright)%7D%3D%5Cfrac%7B1%7D%7B%5Cleft(1%2Be%5E%7B-x%7D%5Cright)%7D%20%5Ccdot%5Cleft(%5Cfrac%7B1%2Be%5E%7B-x%7D%7D%7B1%2Be%5E%7B-x%7D%7D-%5Cfrac%7B1%7D%7B1%2Be%5E%7B-x%7D%7D%5Cright)%0A" alt="
=\frac{1}{\left(1+e^{-x}\right)} \cdot \frac{e^{-x}+1-1}{\left(1+e^{-x}\right)}=\frac{1}{\left(1+e^{-x}\right)} \cdot\left(\frac{1+e^{-x}}{1+e^{-x}}-\frac{1}{1+e^{-x}}\right)
" />

Let’s cancel out the numerator and denominator

<img src="https://tex.s2cms.ru/svg/%0A%3D%5Cfrac%7B1%7D%7B%5Cleft(1%2Be%5E%7B-x%7D%5Cright)%7D%20%5Ccdot%5Cleft(%5Cfrac%7B1%2Be%5E%7B-x%7D%7D%7B1%2Be%5E%7B-x%7D%7D-%5Cfrac%7B1%7D%7B1%2Be%5E%7B-x%7D%7D%5Cright)%3D%5Cfrac%7B1%7D%7B%5Cleft(1%2Be%5E%7B-x%7D%5Cright)%7D%20%5Ccdot%5Cleft(1-%5Cfrac%7B1%7D%7B1%2Be%5E%7B-x%7D%7D%5Cright)%0A" alt="
=\frac{1}{\left(1+e^{-x}\right)} \cdot\left(\frac{1+e^{-x}}{1+e^{-x}}-\frac{1}{1+e^{-x}}\right)=\frac{1}{\left(1+e^{-x}\right)} \cdot\left(1-\frac{1}{1+e^{-x}}\right)
" />

Now, if we take a look at the first equation of this article (1), then we can rewrite as follows

<img src="https://tex.s2cms.ru/svg/%0A%3D%5Cfrac%7B1%7D%7B%5Cleft(1%2Be%5E%7B-x%7D%5Cright)%5E%7B.%7D%7D%20%5Ccdot%5Cleft(1-%5Cfrac%7B1%7D%7B1%2Be%5E%7B-x%7D%7D%5Cright)%3D%5Csigma(x)%20%5Ccdot(1-%5Csigma(x))%0A" alt="
=\frac{1}{\left(1+e^{-x}\right)^{.}} \cdot\left(1-\frac{1}{1+e^{-x}}\right)=\sigma(x) \cdot(1-\sigma(x))
" />


Derivative of the Sigmoid function [towardsdatascience](https://towardsdatascience.com/derivative-of-the-sigmoid-function-536880cf918e):

> Markdown is a lightweight markup language with plain text formatting syntax designed so that it can be converted to HTML and many other formats using a tool by the same name. Markdown is often used to format readme files, for writing messages in online discussion forums, and to create rich text using a plain text editor.

The main idea of Markdown is to use a simple plain text markup. It's ~~hard~~ easy to __make__ **bold** _or_ *italic* text. Simple equations can be formatted with subscripts and superscripts: *E*~0~=*mc*^2^. I have added the LaTeX support: <img src="https://tex.s2cms.ru/svg/E_0%3Dmc%5E2" alt="E_0=mc^2" />.

Among Markdown features are:

* images (see above);
* links: [service main page](/ "link title");
* code: `untouched equation source is *E*~0~=*mc*^2^`;
* unordered lists--when a line starts with `+`, `-`, or `*`;
  1. sub-lists
  1. and ordered lists too;
* direct use <nobr>of HTML</nobr>&ndash;for <span style="color: red">anything else</span>. 

Also the editor supports typographic replacements: (c) (r) (tm) (p) +- !!!!!! ???? ,,  -- ---

## LaTeX

The editor converts LaTeX equations in double-dollars `$$`: <img src="https://tex.s2cms.ru/svg/ax%5E2%2Bbx%2Bc%3D0" alt="ax^2+bx+c=0" />. All equations are rendered as block equations. If you need inline ones, you can add the prefix `\inline`: <img src="https://tex.s2cms.ru/svg/%5Cinline%20p%3D%7B1%5Cover%20q%7D" alt="\inline p={1\over q}" />. But it is a good practice to place big equations on separate lines:

<img src="https://tex.s2cms.ru/svg/x_%7B1%2C2%7D%20%3D%20%7B-b%5Cpm%5Csqrt%7Bb%5E2%20-%204ac%7D%20%5Cover%202a%7D." alt="x_{1,2} = {-b\pm\sqrt{b^2 - 4ac} \over 2a}." />

In this case the LaTeX syntax will be highlighted in the source code. You can even add equation numbers (unfortunately there is no automatic numbering and refs support):

<img src="https://tex.s2cms.ru/svg/%7C%5Cvec%7BA%7D%7C%3D%5Csqrt%7BA_x%5E2%20%2B%20A_y%5E2%20%2B%20A_z%5E2%7D." alt="|\vec{A}|=\sqrt{A_x^2 + A_y^2 + A_z^2}." />(1)

It is possible to write Cyrillic symbols in `\text` command: <img src="https://tex.s2cms.ru/svg/Q_%5Ctext%7B%D0%BF%D0%BB%D0%B0%D0%B2%D0%BB%D0%B5%D0%BD%D0%B8%D1%8F%7D%3E0" alt="Q_\text{плавления}&gt;0" />.

One can use matrices:

<img src="https://tex.s2cms.ru/svg/T%5E%7B%5Cmu%5Cnu%7D%3D%5Cbegin%7Bpmatrix%7D%0A%5Cvarepsilon%260%260%260%5C%5C%0A0%26%5Cvarepsilon%2F3%260%260%5C%5C%0A0%260%26%5Cvarepsilon%2F3%260%5C%5C%0A0%260%260%26%5Cvarepsilon%2F3%0A%5Cend%7Bpmatrix%7D%2C" alt="T^{\mu\nu}=\begin{pmatrix}
\varepsilon&amp;0&amp;0&amp;0\\
0&amp;\varepsilon/3&amp;0&amp;0\\
0&amp;0&amp;\varepsilon/3&amp;0\\
0&amp;0&amp;0&amp;\varepsilon/3
\end{pmatrix}," />

integrals:

<img src="https://tex.s2cms.ru/svg/P_%5Comega%3D%7Bn_%5Comega%5Cover%202%7D%5Chbar%5Comega%5C%2C%7B1%2BR%5Cover%201-v%5E2%7D%5Cint%5Climits_%7B-1%7D%5E%7B1%7Ddx%5C%2C(x-v)%7Cx-v%7C%2C" alt="P_\omega={n_\omega\over 2}\hbar\omega\,{1+R\over 1-v^2}\int\limits_{-1}^{1}dx\,(x-v)|x-v|," />

cool tikz-pictures:

<img src="https://tex.s2cms.ru/svg/%5Cusetikzlibrary%7Bdecorations.pathmorphing%7D%0A%5Cbegin%7Btikzpicture%7D%5Bline%20width%3D0.2mm%2Cscale%3D1.0545%5D%5Csmall%0A%5Ctikzset%7B%3E%3Dstealth%7D%0A%5Ctikzset%7Bsnake%20it%2F.style%3D%7B-%3E%2Csemithick%2C%0Adecoration%3D%7Bsnake%2Camplitude%3D.3mm%2Csegment%20length%3D2.5mm%2Cpost%20length%3D0.9mm%7D%2Cdecorate%7D%7D%0A%5Cdef%5Ch%7B3%7D%0A%5Cdef%5Cd%7B0.2%7D%0A%5Cdef%5Cww%7B1.4%7D%0A%5Cdef%5Cw%7B1%2B%5Cww%7D%0A%5Cdef%5Cp%7B1.5%7D%0A%5Cdef%5Cr%7B0.7%7D%0A%5Ccoordinate%5Blabel%3Dbelow%3A%24A_1%24%5D%20(A1)%20at%20(%5Cww%2C%5Cp)%3B%0A%5Ccoordinate%5Blabel%3Dabove%3A%24B_1%24%5D%20(B1)%20at%20(%5Cww%2C%5Cp%2B%5Ch)%3B%0A%5Ccoordinate%5Blabel%3Dbelow%3A%24A_2%24%5D%20(A2)%20at%20(%5Cw%2C%5Cp)%3B%0A%5Ccoordinate%5Blabel%3Dabove%3A%24B_2%24%5D%20(B2)%20at%20(%5Cw%2C%5Cp%2B%5Ch)%3B%0A%5Ccoordinate%5Blabel%3Dleft%3A%24C%24%5D%20(C1)%20at%20(0%2C0)%3B%0A%5Ccoordinate%5Blabel%3Dleft%3A%24D%24%5D%20(D)%20at%20(0%2C%5Ch)%3B%0A%5Cdraw%5Bfill%3Dblue!14%5D(A2)--(B2)--%20%2B%2B(%5Cd%2C0)--%20%2B%2B(0%2C-%5Ch)--cycle%3B%0A%5Cdraw%5Bgray%2Cthin%5D(C1)--%20%2B(%5Cw%2B%5Cd%2C0)%3B%0A%5Cdraw%5Bdashed%2Cgray%2Cfill%3Dblue!5%5D(A1)--%20(B1)--%20%2B%2B(%5Cd%2C0)--%20%2B%2B(0%2C-%5Ch)--%20cycle%3B%0A%5Cdraw%5Bdashed%2Cline%20width%3D0.14mm%5D(A1)--(C1)--(D)--(B1)%3B%0A%5Cdraw%5Bsnake%20it%5D(C1)--(A2)%20node%5Bpos%3D0.6%2Cbelow%5D%20%7B%24c%5CDelta%20t%24%7D%3B%0A%5Cdraw%5B-%3E%2Csemithick%5D(%5Cww%2C%5Cp%2B0.44*%5Ch)--%20%2B(%5Cw-%5Cww%2C0)%20node%5Bpos%3D0.6%2Cabove%5D%20%7B%24v%5CDelta%20t%24%7D%3B%0A%5Cdraw%5Bsnake%20it%5D(D)--(B2)%3B%0A%5Cdraw%5Bthin%5D(%5Cr%2C0)%20arc%20(0%3Aatan2(%5Cp%2C%5Cw)%3A%5Cr)%20node%5Bmidway%2Cright%2Cyshift%3D0.06cm%5D%20%7B%24%5Ctheta%24%7D%3B%0A%5Cdraw%5Bopacity%3D0%5D(-0.40%2C-0.14)--%20%2B%2B(0%2C5.06)%3B%0A%5Cend%7Btikzpicture%7D" alt="\usetikzlibrary{decorations.pathmorphing}
\begin{tikzpicture}[line width=0.2mm,scale=1.0545]\small
\tikzset{&gt;=stealth}
\tikzset{snake it/.style={-&gt;,semithick,
decoration={snake,amplitude=.3mm,segment length=2.5mm,post length=0.9mm},decorate}}
\def\h{3}
\def\d{0.2}
\def\ww{1.4}
\def\w{1+\ww}
\def\p{1.5}
\def\r{0.7}
\coordinate[label=below:$A_1$] (A1) at (\ww,\p);
\coordinate[label=above:$B_1$] (B1) at (\ww,\p+\h);
\coordinate[label=below:$A_2$] (A2) at (\w,\p);
\coordinate[label=above:$B_2$] (B2) at (\w,\p+\h);
\coordinate[label=left:$C$] (C1) at (0,0);
\coordinate[label=left:$D$] (D) at (0,\h);
\draw[fill=blue!14](A2)--(B2)-- ++(\d,0)-- ++(0,-\h)--cycle;
\draw[gray,thin](C1)-- +(\w+\d,0);
\draw[dashed,gray,fill=blue!5](A1)-- (B1)-- ++(\d,0)-- ++(0,-\h)-- cycle;
\draw[dashed,line width=0.14mm](A1)--(C1)--(D)--(B1);
\draw[snake it](C1)--(A2) node[pos=0.6,below] {$c\Delta t$};
\draw[-&gt;,semithick](\ww,\p+0.44*\h)-- +(\w-\ww,0) node[pos=0.6,above] {$v\Delta t$};
\draw[snake it](D)--(B2);
\draw[thin](\r,0) arc (0:atan2(\p,\w):\r) node[midway,right,yshift=0.06cm] {$\theta$};
\draw[opacity=0](-0.40,-0.14)-- ++(0,5.06);
\end{tikzpicture}" />

plots:

<img src="https://tex.s2cms.ru/svg/%5Cbegin%7Btikzpicture%7D%5Bscale%3D1.0544%5D%5Csmall%0A%5Cbegin%7Baxis%7D%5Baxis%20line%20style%3Dgray%2C%0A%09samples%3D120%2C%0A%09width%3D9.0cm%2Cheight%3D6.4cm%2C%0A%09xmin%3D-1.5%2C%20xmax%3D1.5%2C%0A%09ymin%3D0%2C%20ymax%3D1.8%2C%0A%09restrict%20y%20to%20domain%3D-0.2%3A2%2C%0A%09ytick%3D%7B1%7D%2C%0A%09xtick%3D%7B-1%2C1%7D%2C%0A%09axis%20equal%2C%0A%09axis%20x%20line%3Dcenter%2C%0A%09axis%20y%20line%3Dcenter%2C%0A%09xlabel%3D%24x%24%2Cylabel%3D%24y%24%5D%0A%5Caddplot%5Bred%2Cdomain%3D-2%3A1%2Csemithick%5D%7Bexp(x)%7D%3B%0A%5Caddplot%5Bblack%5D%7Bx%2B1%7D%3B%0A%5Caddplot%5B%5D%20coordinates%20%7B(1%2C1.5)%7D%20node%7B%24y%3Dx%2B1%24%7D%3B%0A%5Caddplot%5Bred%5D%20coordinates%20%7B(-1%2C0.6)%7D%20node%7B%24y%3De%5Ex%24%7D%3B%0A%5Cpath%20(axis%20cs%3A0%2C0)%20node%20%5Banchor%3Dnorth%20west%2Cyshift%3D-0.07cm%5D%20%7B0%7D%3B%0A%5Cend%7Baxis%7D%0A%5Cend%7Btikzpicture%7D" alt="\begin{tikzpicture}[scale=1.0544]\small
\begin{axis}[axis line style=gray,
	samples=120,
	width=9.0cm,height=6.4cm,
	xmin=-1.5, xmax=1.5,
	ymin=0, ymax=1.8,
	restrict y to domain=-0.2:2,
	ytick={1},
	xtick={-1,1},
	axis equal,
	axis x line=center,
	axis y line=center,
	xlabel=$x$,ylabel=$y$]
\addplot[red,domain=-2:1,semithick]{exp(x)};
\addplot[black]{x+1};
\addplot[] coordinates {(1,1.5)} node{$y=x+1$};
\addplot[red] coordinates {(-1,0.6)} node{$y=e^x$};
\path (axis cs:0,0) node [anchor=north west,yshift=-0.07cm] {0};
\end{axis}
\end{tikzpicture}" />

and [the rest of LaTeX features](https://en.wikibooks.org/wiki/LaTeX/Mathematics).


