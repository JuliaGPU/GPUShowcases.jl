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
