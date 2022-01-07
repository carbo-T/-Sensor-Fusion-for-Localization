### 线特征

$$
\pmb{d}_\varepsilon=\frac{(\tilde{p_i}-p_b)\cross(\tilde{p_i}-p_a)}{|p_a-p_b|}
\\线特征残差: d_\varepsilon=\frac{|(\tilde{p_i}-p_b)\cross(\tilde{p_i}-p_a)|}{|p_a-p_b|}=|\pmb{d}_\varepsilon|=\sqrt{\pmb{d}^T_\varepsilon \pmb{d}_\varepsilon}
$$

$$
雅克比：J_\varepsilon=\frac{\partial{d_\varepsilon}}{\partial{T}}=\frac{\partial{d_\varepsilon}}{\partial{\pmb{d}_\varepsilon}}\frac{\partial{\pmb{d}_\varepsilon}}{\partial{\tilde{p_i}}}\frac{\partial{\tilde{p_i}}}{\partial{T}}，其中T=[\begin{matrix} R & t \\ 0^T & 1 \end{matrix}] \in SE(3)
$$

$$
=\frac{\pmb{d}^T_\varepsilon}{d_\varepsilon}
\frac{\partial(\tilde{p_i}-p_b)\cross(\tilde{p_i}-p_a)+(\tilde{p_i}-p_b)\cross{\partial(\tilde{p_i}-p_a)}}{\partial{\tilde{p_i}}|p_a-p_b|}\frac{\partial{\tilde{p_i}}}{\partial{T}}
$$

$$
=\frac{\pmb{d}^T_\varepsilon}{d_\varepsilon}\frac{-(\tilde{p_i}-p_a)_\cross+(\tilde{p_i}-p_b)_\cross}{|p_a-p_b|}\frac{\partial{\tilde{p_i}}}{\partial{T}}
$$

$$
=\frac{\pmb{d}^T_\varepsilon}{d_\varepsilon} \frac{(p_a-p_b)_\cross}{|p_a-p_b|}\frac{\partial{\tilde{p_i}}}{\partial{T}}
$$

#### 李群、李代数与变换矩阵导数计算

$$
推导\frac{\partial{\tilde{p_i}}}{\partial{T}},首先分析旋转矩阵R \in SO(3)
$$

$$
R具有特性RR^T=I,将R看做时间的函数，则\\R(t)R(t)^T=I,\\对时间求导，得\\R^`(t)R(t)^T+R(t)R^`(t)^T=0,即R^`(t)R(t)^T=-(R^`(t)R(t)^T)^T,具有反对称矩阵特性，\\因此可找到\phi(t) \in \R^3,使得\\R^`(t)R(t)^T=\phi(t)_\cross\\右乘R(t),得R^`(t)=\phi(t)_\cross R(t),即表示旋转矩阵求导可当做左乘一个\phi(t)_\cross,\\设R(0)=I,一阶泰勒展开R(t)\approx R(t_0)+R^`(t_0)(t-t_0)=I+\phi(t_0)_\cross (t),\\在原点附近存在R(t)=exp(\phi_{0\cross} t)
$$

$$
以上指数映射可以写成泰勒展开exp(\phi_{0\cross})=\sum^\infty_{n=0}\frac{1}{n}(\phi_\cross)^n,\\由于反对称阵特性，a_\cross a_\cross = aa^T-I,a_\cross a_\cross a_\cross=-a_\cross,\\因此指数映射展开exp(\phi_\cross)=exp(\theta a_\cross),其中a与\theta为轴角旋转表示形式中的旋转轴与角度,\\泰勒展开后可推算出罗德里格斯公式，即exp(\theta a_\cross)=\cos \theta I+(1-\cos \theta )aa^T+\sin \theta a_\cross.\\同理得SE(3)指数映射exp(\xi_\cross)=[\begin{matrix} exp(\phi_\cross) & J\rho \\ 0^T & 1 \end{matrix}],其中J=\frac{\sin(\theta)}{\theta}I+(1-\frac{\sin(\theta)}{\theta})aa^T+\frac{1-\cos(\theta)}{\theta}a_\cross
$$

$$
根据左乘扰动模型和罗德里格斯公式，可以求出对应李代数导数.
\\针对SO(3),\frac{\partial{(Rp)}}{\partial{\varphi}}=\lim_{\varphi\rarr0}\frac{exp(\varphi_\cross)exp(\phi_\cross)p-exp(\phi_\cross)p}{\varphi},其中扰动\Delta R对应李代数为\varphi.
\\根据罗德里格斯公式与\Delta\theta\rarr0条件,可得\frac{\partial{Rp}}{\partial{\varphi}}=-(Rp)_\cross.
\\针对SE(3),\frac{\partial{(Tp)}}{\partial{\delta\xi}}=\lim_{\delta\xi\rarr0}\frac{exp(\delta\xi_\cross)exp(\xi_\cross)p-exp(\xi_\cross)p}{\delta\xi},其中扰动\Delta T对应李代数为\delta\xi=[\delta\rho,\delta\phi]^T.
\\同理可得\frac{\partial{(Tp)}}{\partial{\delta\xi}}=[\begin{matrix} I & -(Rp+t)_\cross \\ 0^T & 0^T \end{matrix}]
$$

$$
因此，\frac{\partial{\tilde{p_i}}}{\partial{T}}=[\begin{matrix} I & -(R\tilde{p_i}+t)_\cross \\ 0^T & 0^T \end{matrix}] \in \R^{4\cross 6},考虑到雅克比前者矩阵大小，调整为[I \space\space -(R\tilde{p_i}+t)_\cross] \in \R^{3\cross6}
$$

$$
雅克比：J_\varepsilon=\frac{\pmb{d}^T_\varepsilon}{d_\varepsilon}\frac{(p_a-p_b)_\cross}{|p_a-p_b|}\frac{\partial{\tilde{p_i}}}{\partial{T}}
\\=[\frac{\pmb{d}^T_\varepsilon}{d_\varepsilon}\frac{(p_a-p_b)_\cross}{|p_a-p_b|} \space\space -\frac{\pmb{d}^T_\varepsilon}{d_\varepsilon}\frac{(p_a-p_b)_\cross(R\tilde{p_i}+t)_\cross}{|p_a-p_b|}] \in \R^{1\cross6}
$$

### 面特征

$$
\pmb{d}_H=(\tilde{p_i}-p_j)\cdot \frac{(p_l-p_j)\cross(p_m-p_j))}{|(p_l-p_j)\cross(p_m-p_j)|}
\\
面特征残差: d_H=|(\tilde{p_i}-p_j)\cdot \frac{(p_l-p_j)\cross(p_m-p_j))}{|(p_l-p_j)\cross(p_m-p_j)|}|=|\pmb{d}_H|
\\雅克比:J_H=\frac{\partial{d_H}}{\partial{T}}=\frac{\partial{d_H}}{\partial{\pmb{d}_H}}\frac{\partial{\pmb{d}_H}}{\partial{\tilde{p_i}}}\frac{\partial{\tilde{p_i}}}{\partial{T}},其中点对位姿求导\frac{\partial{\tilde{p_i}}}{\partial{T}}与线特征一致
\\=\frac{\pmb{d}^T_H}{d_H}\frac{((p_l-p_j)\cross(p_m-p_j))^T}{|(p_l-p_j)\cross(p_m-p_j)|}[I \space\space -(R\tilde{p_i}+t)_\cross]
\\=[\frac{\pmb{d}^T_H}{d_H}\frac{((p_l-p_j)\cross(p_m-p_j))^T}{|(p_l-p_j)\cross(p_m-p_j)|} \space\space -\frac{\pmb{d}^T_H}{d_H}\frac{((p_l-p_j)\cross(p_m-p_j))^T}{|(p_l-p_j)\cross(p_m-p_j)|}|(R\tilde{p_i}+t)_\cross] \in \R^{1\cross6}
$$

