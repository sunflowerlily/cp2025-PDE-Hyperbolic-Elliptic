考虑弦长度为$L$的钢琴，开始处于静止状态. 在$t=0$时刻, 弦被距离弦末端的琴锤击打:

![image.png](attachment:d2f3d788-9f08-4c19-bde0-9888c3adfa29:image.png)

琴弦在除了固定的末端$x=0$和$x=L$以外的地方会产生振动.

1. 写程序使用有限差分法解方程描述的齐次一阶方程，取$v=100\,\mathrm{m~s}^{-1}$, 初始条件为$\phi(x)=0$, 初始速度$\psi(x)$不为零，其轮廓为

$$
\psi(x) = C {x(L-x)\over L^2} \exp \biggl[ -{(x-d)^2\over2\sigma^2} \biggr],
$$

其中$L=1\,$m, $d=10\,$cm, $C=1\,\mathrm{m~s}^{-1}$, $\sigma=0.3\,$m. 你需要选择一个合适的时间步长$h$. 一个合理的建议是取$h=10^{-6}\,$s.
2. 做一个钢琴琴弦运动的动画.

### 参考答案

### 解:

$u_{i,j+1}=c(u_{i+1,j}+u_{i-1,j})+2(1-c)*u_{i,j}-u_{i,j-1}$
其中$c=a^2\frac{(\Delta t)^2}{(\Delta x)^2}$, 当$c<1$时稳定.

$j=1$时刻的$u$的值为
$u_{i,1}=\frac{c}{2}(u_{i+1,0}+u_{i-1,0})+(1-c)u_{i,0}+u_t(x,0)\Delta t.$
其中$u_t(x,0)$是$t=0$时刻$u$对时间的一阶导数.

```python
a = 100 #m/s
L = 1 #m
d = 0.1 #m
C = 1 #m/s
sigma = 0.3 #m
def u_t(x,C=1,d=0.1,sigma=0.3,L=1):
    return C*x*(L-x)/L/L*np.exp(-(x-d)**2/2/sigma**2)
dx = 0.01
dt = 5e-5
x = np.arange(0,L+dx,dx)
t = np.arange(0,0.1+dt,dt)
u = np.zeros((x.size,t.size),float)
c = (a*dt/dx)**2
print(c)
u[1:-1,1] = c/2*(u[2:,0]+u[:-2,0])+(1-c)*u[1:-1,0]+u_t(x[1:-1])*dt
for j in range(1,t.size-1):
    u[1:-1,j+1] = c*(u[2:,j]+u[:-2,j])+2*(1-c)*u[1:-1,j]-u[1:-1,j-1]
from matplotlib.animation import FuncAnimation
fig = plt.figure()
plt.axis([0,1,u.min()*1.1,u.max()*1.1])
myline, = plt.plot([],[],'g-',lw=2)
def update(j):
    myline.set_data(x,u[:,j])
    return myline,
animation = FuncAnimation(fig,update,
                          frames = t.size, 
                          interval = 1)
```