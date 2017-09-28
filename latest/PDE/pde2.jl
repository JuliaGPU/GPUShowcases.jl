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
