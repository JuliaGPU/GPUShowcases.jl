# PDE

Some PDE algorithms visualized and ported for the GPU.

# PDE 1



```Julia
using CLArrays, GLVisualize, GeometryTypes, GLAbstraction, StaticArrays

TY = Float32
N = 1024
const h    = TY(2*π/N)
const epsn = TY(h * .5)
const C    = TY(2/epsn)
const tau  = TY(epsn * h)
Tfinal = 50.

S(x,y) = exp(-x^2/0.1f0)*exp(-y^2/0.1f0)

ArrayType = CLArray
# real-space and reciprocal-space grids
# the real-space grid is just used for plotting!
X_cpu = convert.(TY, collect(linspace(-pi+h, pi, N)) .* ones(1,N))
X = ArrayType(X_cpu);
k = collect([0:N/2; -N/2+1:-1]);
Â = ArrayType(convert.(TY,kron(k.^2, ones(1,N)) + kron(ones(N), k'.^2)));

# initial condition
uc = ArrayType(TY(2.0)*(rand(TY, N, N)-TY(0.5)))

#################################################################
#################################################################

pow3(u) = complex((u * u * u) - u)
function take_step!(u, Â, t_plot, fftplan!, ifftplan!, u3fft, uc, tmp)
    u3fft .= pow3.(u)
    fftplan! * u3fft
    uc .= complex.(u)
    fftplan! * uc
    @. tmp .= ((1f0+C*tau*Â) .* uc .- tau/epsn * (Â .* u3fft)) ./ (1f0+(epsn*tau)*Â.^2f0+C*tau*Â)
    ifftplan! * tmp
    u .= real.(tmp)
    nothing
end
function normalise_af!(u, out)
    out .= u .- minimum(u)
    out .= out ./ maximum(out)
    nothing
end
#################################################################
#################################################################

n = 1
T_plot = 0.01; t_plot = 0.0
ceil(Tfinal / tau)
up = copy(uc)
ucc = complex.(uc)
fftplan! = plan_fft!(ucc)
ifftplan! = plan_ifft!(ucc)
u3fft = similar(ucc)
tmp = similar(ucc)

w = glscreen(resolution = (N, N))
normalise_af!(uc,up)
up .= up .* 0.1f0
robj = visualize(
    reinterpret(Intensity{Float32}, Array(up)),
    stroke_width = 0f0,
    levels = 20f0,
    color_map = Colors.colormap("RdBu", 100),
    color_norm = Vec2f0(0, 0.1)
).children[]
_view(robj, position = Vec3f0(0, 0.5, 2.5), lookat = Vec3f0(0))
io, buffer = GLVisualize.create_video_stream(homedir()*"/Desktop/pd1.mkv", w)
center!(w, :orthographic_pixel)
idx = 0
while isopen(w)
    take_step!(uc, Â, t_plot, fftplan!, ifftplan!, u3fft, ucc, tmp)
    t_plot += tau
    normalise_af!(uc, up)
    up .= up .* 0.1f0
    GLAbstraction.update!(robj[:intensity], reinterpret(SVector{1, Float32}, Array(up)))
    GLWindow.poll_glfw()
    GLWindow.reactive_run_till_now()
    GLWindow.render_frame(w)
    GLWindow.swapbuffers(w)
    if idx == 4
        GLVisualize.add_frame!(io, w, buffer)
        idx = 0
    end
    idx += 1
end
close(io)
GLWindow.destroy!(w)
```

```@raw html
<video width="100%" controls>
  <source src="pde1.webm" type="video/webm">
  Your browser does not support webm. Please use a modern browser like Chrome or Firefox.
</video>
```


# PDE 2

Show case ported from:

[Kuramoto-Sivashinksy-benchmark](https://github.com/johnfgibson/julia-pde-benchmark/blob/master/1-Kuramoto-Sivashinksy-benchmark.ipynb)

```Julia
using CLArrays, GLVisualize, GPUArrays, GLAbstraction, GeometryTypes

# source: https://github.com/johnfgibson/julia-pde-benchmark/blob/master/1-Kuramoto-Sivashinksy-benchmark.ipynb
function inner_ks(n, IFFT!, FFT!, Nt, Nn, Nn1, u, G, A_inv, B, dt2, dt32, uslice, U)
    Nn1 .= Nn       # shift nonlinear term in time
    Nn .= u         # put u into Nn in prep for comp of nonlinear term

    IFFT! * Nn

    # plotting
    uslice .= real.(Nn) ./ 10f0
    U[1:Nt, n] = reshape(Array(uslice), (Nt, 1)) # copy from gpu to opengl gpu not implemented for now

    # transform Nn to gridpt values, in place
    Nn .= Nn .* Nn   # collocation calculation of u^2
    FFT!*Nn        # transform Nn back to spectral coeffs, in place

    Nn .= G .* Nn    # compute Nn == -1/2 d/dx (u^2) = -u u_x

    # loop fusion! Julia translates the folling line of code to a single for loop.
    u .= A_inv .* (B .* u .+ dt32 .* Nn .- dt2 .* Nn1)
end

T = Float32; AT = CLArray
N = 1000
Lx = T(64*pi)
Nx = T(N)
dt = T(1/16)

x = Lx*(0:Nx-1)/Nx
u = T.(cos.(x) + 0.1*sin.(x/8) + 0.01*cos.((2*pi/Lx)*x))

u = AT((T(1)+T(0)im)*u)             # force u to be complex
Nx = length(u)                      # number of gridpoints
kx = T.(vcat(0:Nx/2-1, 0:0, -Nx/2+1:-1))# integer wavenumbers: exp(2*pi*kx*x/L)
alpha = T(2)*pi*kx/Lx                  # real wavenumbers:    exp(alpha*x)

D = T(1)im*alpha                       # spectral D = d/dx operator

L = alpha.^2 .- alpha.^4            # spectral L = -D^2 - D^4 operator

G = AT(T(-0.5) .* D)               # spectral -1/2 D operator, to eval -u u_x = 1/2 d/dx u^2

# convenience variables
dt2  = T(dt/2)
dt32 = T(3*dt/2)
A_inv = AT((ones(T, Nx) - dt2*L).^(-1))
B = AT(ones(T, Nx) + dt2*L)

# compute in-place FFTW plans
FFT! = plan_fft!(u)
IFFT! = plan_ifft!(u)

# compute nonlinear term Nn == -u u_x
powed = u .* u
Nn = G .* fft(powed);    # Nn == -1/2 d/dx (u^2) = -u u_x
Nn1 = copy(Nn);        # Nn1 = Nn at first time step
FFT! * u;

uslice = real(u)
U = zeros(Float32, N, N)

w = glscreen(resolution = (1920, 1080))

robj = visualize(
    U, :surface, color_norm = Vec2f0(-0.1, 0.1),
    ranges = ((-3f0, 3f0), (-3f0, 3f0))
).children[]
Ugpu = robj[:position_z]

# setup camera and view object
_view(robj, position = Vec3f0(-4.33, 3.8, -2.7), lookat = Vec3f0(-0.23, 0.5, 0.7))
cam = w.cameras[:perspective]
push!(cam.up, Vec3f0(0.3, -0.33, -0.9))

io, buffer = GLVisualize.create_video_stream(homedir()*"/Desktop/pd2.mkv", w)
for n in 1:N
    isopen(w) || break
    inner_ks(n, (IFFT!), (FFT!), N, Nn, Nn1, u, G, A_inv, B, dt2, dt32, uslice, Ugpu)
    GLWindow.poll_glfw()
    GLWindow.reactive_run_till_now()
    GLWindow.render_frame(w)
    GLWindow.swapbuffers(w)
    (n % 4 == 0) && GLVisualize.add_frame!(io, w, buffer)
end
close(io)
GLWindow.destroy!(w)
```

```@raw html
<video width="100%" controls>
  <source src="pde2.webm" type="video/webm">
  Your browser does not support the video tag. Please use a modern browser like Chrome or Firefox.
</video>
```
