using SchroedingersSmoke, CLArrays
using Colors, GPUArrays, GeometryTypes, GLAbstraction

ArrayType = CLArray

vol_size = (4,2,2)# box size
dims = (64, 32, 32) .* 2 # volume resolution
hbar = 0.1f0      # Planck constant
dt = 1f0/48f0     # time step

jet_velocity = (1f0, 0f0, 0f0)
nozzle_cen = Float32.((2-1.7, 1-0.034, 1+0.066))
nozzle_len = 0.5f0
nozzle_rad = 0.5f0
n_particles = 500   # number of particles

isf = ISF{ArrayType, UInt32, Float32}(vol_size, dims, hbar, dt);

# initialize psi
psi = ArrayType([(one(Complex64), one(Complex64) * 0.01f0) for i=1:dims[1], j=1:dims[2], k=1:dims[3]]);

psi .= normalize_psi.(psi);

kvec = jet_velocity ./ hbar;
omega = sum(jet_velocity.^2f0) / (2f0*hbar);
isjetarr = isjet.(isf.positions, (nozzle_cen,), (nozzle_len,), (nozzle_rad,))
# constrain velocity
for iter = 1:10
    restrict_velocity!(isf, psi, kvec, isjetarr, 0f0)
    pressure_project!(isf, psi)
end

particles = ArrayType(map(x-> (0f0, 0f0, 0f0), 1:(10^6)))

add_particles!(particles, 1:n_particles, nozzle_cen, nozzle_rad)


using GLVisualize;
w = glscreen(color = RGBA(0f0, 0f0, 0f0, 0f0), resolution = (1920, 1080));
lincolor = RGBA{Float32}(0.0f0,0.74736935f0,1.0f0,0.1f0)
offset = translationmatrix(Vec3f0(0))
dircolor = map(x-> RGBA{Float32}(x[1], x[2], x[3], 0.1), isf.velocity);
dircolorcpu = Array{RGBA{Float32}}(size(isf.velocity))
copy!(dircolorcpu, dircolor)
dircolorviz = visualize(
    dircolorcpu, :absorption, model = offset,
    dimensions = Vec3f0(vol_size)
).children[]
# first view sets up camera
_view(dircolorviz, camera = :perspective, position = Vec3f0(4, -7, 2.1), lookat = Vec3f0(4, 0, 2))

_view(visualize(
    AABB(Vec3f0(0), Vec3f0(vol_size)), :lines,
    color =	lincolor,
    model = offset,
), camera = :perspective)
_view(visualize(
    "Velocity as Colors",
    start_position = Point3f0(0), billboard = false,
    model = translationmatrix(Vec3f0(0.1, -0.1, -0.2)) * rotationmatrix_x(Float32(0.5pi)),
    relative_scale = 0.2f0, color = RGBA(1f0,1f0,1f0,1f0)
), camera = :perspective)
offset = translationmatrix(Vec3f0(0, 0, 2))

particlescpu = Array(particles)
particle_vis = visualize(
    (GeometryTypes.Circle(GeometryTypes.Point2f0(0), 0.006f0), reinterpret(Point3f0, particlescpu)),
    boundingbox = nothing, # don't waste time on bb computation
    model = offset,
    color = fill(RGBA{Float32}(0, 0, 0, 0.09), length(particles)),
    billboard = true
).children[]
_view(particle_vis, camera = :perspective)
_view(visualize(
    AABB(Vec3f0(0), Vec3f0(vol_size)), :lines,
    color =	lincolor,
    model = offset,
), camera = :perspective)

particle_vis[:color][1:n_particles] = map(1:n_particles) do i
    xx = (i / n_particles) * 2pi
    RGBA{Float32}((sin(xx) + 1) / 2, (cos(xx) + 1.0) / 2.0, 0.0, 0.1)
end
_view(visualize(
    "Smoke Particle",
    start_position = Point3f0(0), billboard = false,
    model = translationmatrix(Vec3f0(0.1, -0.1, 4.1)) * rotationmatrix_x(Float32(0.5pi)),
    relative_scale = 0.2f0, color = RGBA(1f0,1f0,1f0,1f0)
), camera = :perspective)




offset = translationmatrix(Vec3f0(4, 0, 0))
magnitude = map(x->sqrt(sum(x .* x)), isf.velocity);
magnitudecpu = Array(magnitude)
magnitudeviz = visualize(
    magnitudecpu, :mip,
    model = offset,
    color_map = [RGBA(0f0, 0f0, 0f0, 0f0), RGBA(0.2f0, 0f0, 0.9f0, 0.9f0), RGBA(0.9f0, 0.2f0, 0f0, 0f0)],
    dimensions = Vec3f0(vol_size)
).children[]
_view(magnitudeviz, camera = :perspective)
_view(visualize(
    AABB(Vec3f0(0), Vec3f0(vol_size)), :lines,
    model = offset,
    color =	lincolor,
), camera = :perspective)
_view(visualize(
    "Velocity Magnitude",
    start_position = Point3f0(0), billboard = false,
    model = translationmatrix(Vec3f0(4.1, -0.1, -0.2)) * rotationmatrix_x(Float32(0.5pi)),
    relative_scale = 0.2f0, color = RGBA(1f0,1f0,1f0,1f0)
), camera = :perspective)

offset = translationmatrix(Vec3f0(4, 0, 2))
directions = copy(isf.velocity)
directionscpu = Array(directions)
directionsviz = visualize(
    reinterpret(Vec3f0, directionscpu),
    model = offset,
    color_map = [RGBA(0f0, 0f0, 0f0, 0f0), RGBA(0.2f0, 0f0, 0.9f0, 0.9f0), RGBA(0.9f0, 0.2f0, 0f0, 0f0)],
    ranges = map(x-> 0:x, vol_size)
).children[]
_view(directionsviz, camera = :perspective)
_view(visualize(
    "Velocity as Vectorfield",
    start_position = Point3f0(0), billboard = false,
    model = translationmatrix(Vec3f0(4.1, -0.1, 4.1)) * rotationmatrix_x(Float32(0.5pi)),
    relative_scale = 0.2f0, color = RGBA(1f0,1f0,1f0,1f0)
), camera = :perspective)

_view(visualize(
    AABB(Vec3f0(0), Vec3f0(vol_size)), :lines,
    model = offset,
    color =	lincolor,
), camera = :perspective)

dt = isf.dt; d = isf.d
iter = 1

io, buffer = GLVisualize.create_video_stream("test.mkv", w)

# main simulation loop
while isopen(w)
    t = iter * dt
    # incompressible Schroedinger flow
    schroedinger_flow!(isf, psi)
    psi .= normalize_psi.(psi)
    pressure_project!(isf, psi)

    start = mod((iter - 1) * n_particles + 1, length(particles))
    stop = start + n_particles - 1
    add_particles!(particles, start:stop, nozzle_cen, nozzle_rad)
    particle_vis[:color][start:stop] = map(1:n_particles) do i
        xx = (i / n_particles) * 2pi
        RGBA{Float32}((sin(xx) + 1) / 2, (cos(xx) + 1.0) / 2.0, mod(iter, 100) / 100, 0.1)
    end
    #constrain velocity
    restrict_velocity!(isf, psi, kvec, isjetarr, omega*t)
    pressure_project!(isf, psi)
    velocity_one_form!(isf, psi, isf.hbar)
    # inplace StaggeredSharp
    dinv = inv.(isf.d)
    broadcast!((x, y)-> x .* y, isf.velocity, isf.velocity, (dinv,))
    staggered_advect!(particles, isf)

    copy!(particlescpu, particles)
    GLAbstraction.set_arg!(particle_vis, :position, reinterpret(Point3f0, particlescpu))

    dircolor .= (x-> RGBA{Float32}(x[1] * 10f0, x[2] * 10f0, x[3] * 10f0, 1.0)).(isf.velocity)
    copy!(dircolorcpu, dircolor)
    GLAbstraction.set_arg!(dircolorviz, :volumedata, dircolorcpu)
    magnitude .= (x->sqrt(sum(x .* x))).(isf.velocity)
    copy!(magnitudecpu, magnitude)
    GLAbstraction.set_arg!(magnitudeviz, :volumedata, magnitudecpu)

    copy!(directionscpu, isf.velocity)
    GLAbstraction.set_arg!(directionsviz, :rotation, vec(reinterpret(Vec3f0, directionscpu)))
    GLWindow.poll_glfw()
    GLWindow.reactive_run_till_now()
    GLWindow.render_frame(w)
    GLWindow.swapbuffers(w)
    GLVisualize.add_frame!(io, w, buffer)
    iter += 1
end
close(io)
GLWindow.destroy!(w)
pwd()
