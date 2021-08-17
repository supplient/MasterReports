目前，为了实现基于形变的破坏检测，采取的方法是计算每个簇的形变程度，如果形变程度很大就认为它发生了破坏。

这之中最重要的一步就是计算“簇的形变程度”。

目前的方案概括来说就是将“簇的最大拉伸程度”作为其形变程度。例如一个物体沿x轴放大4倍，沿y轴缩小8倍，那我们就认为它的形变程度是$max(4, 8) = 8$。计算过程见附录A，概括来说就是对所有粒子形变后位移和形变前位移的零均值协方差进行SVD分解，其最大特征值就是最大拉伸程度。

该方案的问题是柔体仅仅只是在重力作用下也会发生形变，我们将只有重力作用下、并只和地面发生碰撞的情况下物体的最大形变程度称为“自然最大形变程度”。那么显然我们不能将破坏检测的阈值设得比这个“自然最大形变程度”还要小，不然的话物体就算没有和切割工具发生碰撞，我们也会以为它的形变是由切割工具引起的。
* 举个例子来说，刚刚跑的一个实验里，其自然最大形变程度为1.9447，那么我们就不能将破坏检测的阈值设得比这个值还要小，至少也要设到2.0左右才行。

但问题是物体和切割工具的碰撞是否一定引起比自然形变更剧烈的形变？在切割工具相对于簇而言比较细小的时候，答案是否定的。因为此时和切割工具发生碰撞的仅仅只是簇中的很少一部分粒子，它虽然会引起局部的剧烈形变，但是依然比不上能作用在所有粒子上的重力对整个簇形变程度的影响。
* 另一方面，当局部剧烈形变强到能超过重力带来的形变时，那因为这个局部的形变实在是太夸张了，从而就会引起粒子隧穿(tunneling)（也就是粒子从切割工具的一侧直接跑到它的另一侧去）。
* 现实世界中，物体被破坏确确实实就是由局部极端形变导致的，但是目前我们的solver似乎并不具备支持极端形变的稳定性。

该问题的本质就是形变阈值无法设定为一个恰当的数值来区分自然形变和碰撞形变。
* 如下面的gif，因为形变阈值设得过大，所以哪怕与切割工具的碰撞引起了形变也没有被切开。在局部形变剧烈到能够触及阈值前就先因为粒子隧穿问题而穿过了切割工具。![](/ContactCut探索/基于簇的形变的问题_阈值过大.gif)
* 而当形变阈值过小时，自然形变也会高过该阈值，使得该破坏检测过程退化为基于距离的破坏检测。如该gif最后部分，躺在地上的那一块儿很明显没有和切割工具发生碰撞但还是被切开了：![](/ContactCut探索/基于簇的形变的问题_阈值过小.gif)

有几个方案能缓解该问题：
* 可以通过调小簇的大小来缓解该问题，让由切割工具引起的局部形变尽可能接近簇的整体形变即可，但那样的话物体刚性会变弱，而且因为此时需要更多簇了，所以计算量也会随之增大。
* 可以通过调小物体刚性来缓解问题。之所以能缓解是因为shape-matching约束越弱就越不容易引起粒子隧穿问题，但问题是此时重力引起的自然形变程度也会增大，所以终究只是缓解一点，并不能解决问题。而且我们不希望我们的方法只适用于软趴趴的物体。


目前的一个想法是不使用“基于簇的形变”，而是采用更局部的“基于粒子的形变”。先看这张图：
![](/ContactCut探索/看似完美的碰撞形变.PNG)

图中是物体与切割工具发生碰撞，然后被挤压出来一条沟。此时因为簇的形变程度都比自然最大形变程度要小，所以可以看到沟的左右两边依然有mesh相连，物体没有被切开。

但是虽然mesh没有被切开，粒子也确确实实被左右分割开来了。这给我的提示就是可以使用“到邻居粒子的距离”作为“形变程度”的判断依据。
* 也就是说，当一个粒子距离其某一个邻居非常远时，就认为在它和它邻居之间发生了破坏。

下一步打算尝试这种“基于粒子的形变”。










# 附录A. 簇最大拉伸程度的计算
【这部分基本只是我自己备忘而已】

我们每个阶段可以从flex那里得到各粒子的deformed positions（形变后位移），并且对柔体建模的时候我们就存下来各粒子的rest positions（形变前位移）。利用这两个量就足以判定簇的形变程度了，下面进行推导。

## Defination
Let the *i*th particle's deformed position to be $x^*_i$(column vector), and its rest position to be $r^*_i$

Use $\overline{x^*}$ for the means of $x^*_i$: $\overline{x^*} = \frac{1}{n} \sum_i x^*_i$.
And the same for $\overline{r^*} = \frac{1}{n} \sum_i r^*_i$

Let $x_i = x^*_i - \overline{x^*}$, and $x = \begin{bmatrix}
	x_1, x_2, \cdots, x_n
\end{bmatrix}$. And the same for $r_i, r$

Now $x, r$ are zero-mean random variables. This feature will be used later.


## Goal Position
The magnitude of deformation using shape-matching method can be define as the stretch extent between the deformed position and the goal position.

In shape-matching method, the goal position is computed as(refer to literature 1):

$$
	g_i = Rr_i
$$

$R$ is the result of polar decomposition on the best transformation $A = RS$. And $A$ is computed by minimize the least square sum:

$$
\min_A \sum_i m_i(Ar_i - x_i)^2
$$

where $m_i$ is the mass of *i*th particle. Derivates the expression:

$$
\begin{aligned}
	\frac{\partial(\sum_i m_i(Ar_i - x_i)^2)}{\partial A} \\
	= \sum_i 2m_i (Ar_i - x_i)r_i^T
\end{aligned}
$$

Let it to be zero, we can solve $A$:

$$
\begin{aligned}
	A &= \sum_i(x_ir_i^T)(\sum_i(r_ir_i^T))^{-1} \\
	&= xr^T(rr^T)^{-1}
\end{aligned}
$$

Note, $x, r$ are zero-based random variables, so the covariance between them is:

$$
	Cov(x, r) = \sum_i(x_ir_i^T)
$$

And the variance of $r$ is:
$$
	Cov(r, r) = \sum_i(r_i r_i^T)
$$

So $A$ in another form can be:

$$
	A = Cov(x, r) Cov(r,r)^{-1}
$$

This is why the literature 2 uses the covariance matrix for the calcualtion of best rotation. The using of 'covariance matrix' may be ambiguous, since the covariance matrix always refer to the matrix of covariances between two sets of random variables. However, in literature 2, they are just using the covariance between $x$ and $r$, which is a matrix.


## The magnitude of stretch
Now we have the best transformation $A$ containing rotation and scaling.

Refer to the literature 3 and [my blog](https://zhuanlan.zhihu.com/p/397600286), we know the singular values of $A$ are the magnitudes of stretch and the right-singular vectors are the stretch directions.

## Degenerate Condition: $r_i r_i^T$ is a zero matrix
If $r_i r_i^T$ is a zero matrix, the cluster is just a point. So of course there is no deformation.


## 参考文献
1. 2005 Meshless Deformations Based on Shape Matching
2. 2014 Unified particle physics for real-time applications
3. 2016 Ductile Fracture for Clustered Shape Matching