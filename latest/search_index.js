var documenterSearchIndex = {"docs": [

{
    "location": "index.html#",
    "page": "Home",
    "title": "Home",
    "category": "page",
    "text": ""
},

{
    "location": "Convolution/convolution.html#",
    "page": "Convolution",
    "title": "Convolution",
    "category": "page",
    "text": ""
},

{
    "location": "Convolution/convolution.html#Convolution-1",
    "page": "Convolution",
    "title": "Convolution",
    "category": "section",
    "text": ""
},

{
    "location": "PDE/pde.html#",
    "page": "PDE 1",
    "title": "PDE 1",
    "category": "page",
    "text": ""
},

{
    "location": "PDE/pde.html#PDE-1-1",
    "page": "PDE 1",
    "title": "PDE 1",
    "category": "section",
    "text": "Show case ported from:Kuramoto-Sivashinksy-benchmark  <video width=\"100%\" controls>\n  <source src=\"pde1.webm\" type=\"video/webm\">\n  Your browser does not support webm. Please use a modern browser like Chrome or Firefox.\n  </video>"
},

{
    "location": "PDE/pde.html#PDE-2-1",
    "page": "PDE 1",
    "title": "PDE 2",
    "category": "section",
    "text": "  <video width=\"100%\" controls>\n  <source src=\"pde2.webm\" type=\"video/webm\">\n  Your browser does not support the video tag. Please use a modern browser like Chrome or Firefox.\n  </video>"
},

{
    "location": "Poincare/poincare.html#",
    "page": "Poincare",
    "title": "Poincare",
    "category": "page",
    "text": ""
},

{
    "location": "Poincare/poincare.html#Poincare-1",
    "page": "Poincare",
    "title": "Poincare",
    "category": "section",
    "text": "Poincare section of a chaotic neuronal network.Original poincare implementation was written by Raner Engelken It was ported to the GPU by Simon Danischusing CLArrays, GPUArrays\nusing FileIO, Interpolations, Colors, ColorVectorSpace, FixedPointNumbers\n\nfunction poincare_inner{N}(rv, result, c, π, ::Val{N}, n)\n    # find next spiking neuron\n    ϕ₁, ϕ₂, ϕ₃ = rv[1], rv[2], rv[3]\n    πh = π / 2f0\n    π2 = π * 2f0\n    for unused = 1:N\n        if ϕ₁ > ϕ₂\n            if ϕ₁ > ϕ₃\n                # first neuron is spiking\n                dt = πh - ϕ₁\n                # evolve phases till next spike time\n                ϕ₁ = -πh\n                ϕ₂ = atan(tan(ϕ₂ + dt) - c)\n                ϕ₃ += dt\n                # save state of neuron 2 and 3\n                x = Cuint(max(round(((ϕ₂ + πh) / π) * (Float32(n) - 1f0)) + 1f0, 1f0))\n                y = Cuint(max(round(((ϕ₃ + πh) / π) * (Float32(n) - 1f0)) + 1f0, 1f0))\n                accum = result[x, y]\n                # this is unsafe, since it could read + write from different threads, but good enough for the stochastic kind of process we're doing\n                result[i1d] = accum + 1f0\n                continue\n            end\n        else\n            if ϕ₂ > ϕ₃\n                # second neuron is spiking\n                dt = πh - ϕ₂\n                # evolve phases till next spike time\n                ϕ₁ += dt\n                ϕ₂ = -πh\n                ϕ₃ = atan(tan(ϕ₃ + dt) - c)\n                continue\n            end\n        end\n        # third neuron is spikinga\n        dt = πh - ϕ₃\n        # evolve phases till next spike time\n        ϕ₁ += dt\n        ϕ₂ = atan(tan(ϕ₂ + dt) - c)\n        ϕ₃ = -πh\n    end\n    return\nend\n\nfunction poincare_inner(n, seeds::GPUArray, result, c, π, val::Val{N}) where N\n    foreach(poincare_inner, seeds, result, c, Float32(pi), val, n)\nend\n\nc = 1f0; divisor = 2^10\nsrand(2)\nN = 10^10\nND = Cuint(2048)\nAT = CLArray\nresult = AT(zeros(Float32, ND, ND))\n_n = div(N, divisor)\njl_seeds = [ntuple(i-> rand(Float32), Val{3}) for x in 1:divisor]\nseeds = AT(jl_seeds)\npoincare_inner(ND, seeds, Base.RefValue(result), c, Float32(pi), Val{_n}())\n\ncmap = interpolate(([\n    RGB(0.0, 0.0, 0),\n    RGB(0.2, 0.2, 0.9),\n    RGB(0.2, 0.6, 0.9),\n    RGB(0.7, 0.7, 0.98),\n    RGB(0.8, 0.8, 0.9),\n    RGB(0.82, 0.8, 1.0)\n]), BSpline(Linear()), OnCell())\n\ncn = length(cmap)\nresultcpu = Array(result)\nextrema(log.(resultcpu))\n\nimg_color = map(resultcpu) do val\n    val = maxi - val\n    if val ≈ 0.0\n        val = 0.01\n    end\n    val = log(val)\n    val = clamp(val, 0f0, 1f0);\n    idx = (val * (cn - 1)) + 1.0\n    RGB{N0f8}(cmap[idx])\nend\n#save as an image\nsave(joinpath(@__DIR__, \"poincare.png\"), img_color)Running the code results in this pretty picture:(Image: )"
},

{
    "location": "SmokeSimulation/smokesimulation.html#",
    "page": "Smoke Simulation",
    "title": "Smoke Simulation",
    "category": "page",
    "text": ""
},

{
    "location": "SmokeSimulation/smokesimulation.html#Smoke-Simulation-1",
    "page": "Smoke Simulation",
    "title": "Smoke Simulation",
    "category": "section",
    "text": "A simulation running on the GPU using SchroedingersSmoke.jl:using SchroedingersSmoke, CLArrays\nusing Colors, GPUArrays, GeometryTypes, GLAbstraction\n\n# can be any (GPU) Array type supporting the GPU Array interface and the needed intrinsics\n# like sin, etc\n\nArrayType = CLArray\n\nvol_size = (4,2,2)# box size\ndims = (64, 32, 32) .* 2 # volume resolution\nhbar = 0.1f0      # Planck constant\ndt = 1f0/48f0     # time step\n\njet_velocity = (1f0, 0f0, 0f0)\nnozzle_cen = Float32.((2-1.7, 1-0.034, 1+0.066))\nnozzle_len = 0.5f0\nnozzle_rad = 0.5f0\nn_particles = 500   # number of particles\n\nisf = ISF{ArrayType, UInt32, Float32}(vol_size, dims, hbar, dt);\n\n# initialize psi\npsi = ArrayType([(one(Complex64), one(Complex64) * 0.01f0) for i=1:dims[1], j=1:dims[2], k=1:dims[3]]);\n\npsi .= normalize_psi.(psi);\n\nkvec = jet_velocity ./ hbar;\nomega = sum(jet_velocity.^2f0) / (2f0*hbar);\nisjetarr = isjet.(isf.positions, (nozzle_cen,), (nozzle_len,), (nozzle_rad,))\n# constrain velocity\nfor iter = 1:10\n    restrict_velocity!(isf, psi, kvec, isjetarr, 0f0)\n    pressure_project!(isf, psi)\nend\n\nparticles = ArrayType(map(x-> (0f0, 0f0, 0f0), 1:(10^6)))\n\nadd_particles!(particles, 1:n_particles, nozzle_cen, nozzle_rad)\n\n\nusing GLVisualize;\nw = glscreen(color = RGBA(0f0, 0f0, 0f0, 0f0), resolution = (1920, 1080));\nlincolor = RGBA{Float32}(0.0f0,0.74736935f0,1.0f0,0.1f0)\noffset = translationmatrix(Vec3f0(0))\ndircolor = map(x-> RGBA{Float32}(x[1], x[2], x[3], 0.1), isf.velocity);\ndircolorcpu = Array{RGBA{Float32}}(size(isf.velocity))\ncopy!(dircolorcpu, dircolor)\ndircolorviz = visualize(\n    dircolorcpu, :absorption, model = offset,\n    dimensions = Vec3f0(vol_size)\n).children[]\n# first view sets up camera\n_view(dircolorviz, camera = :perspective, position = Vec3f0(4, -7, 2.1), lookat = Vec3f0(4, 0, 2))\n\n_view(visualize(\n    AABB(Vec3f0(0), Vec3f0(vol_size)), :lines,\n    color =	lincolor,\n    model = offset,\n), camera = :perspective)\n_view(visualize(\n    \"Velocity as Colors\",\n    start_position = Point3f0(0), billboard = false,\n    model = translationmatrix(Vec3f0(0.1, -0.1, -0.2)) * rotationmatrix_x(Float32(0.5pi)),\n    relative_scale = 0.2f0, color = RGBA(1f0,1f0,1f0,1f0)\n), camera = :perspective)\noffset = translationmatrix(Vec3f0(0, 0, 2))\n\nparticlescpu = Array(particles)\nparticle_vis = visualize(\n    (GeometryTypes.Circle(GeometryTypes.Point2f0(0), 0.006f0), reinterpret(Point3f0, particlescpu)),\n    boundingbox = nothing, # don't waste time on bb computation\n    model = offset,\n    color = fill(RGBA{Float32}(0, 0, 0, 0.09), length(particles)),\n    billboard = true\n).children[]\n_view(particle_vis, camera = :perspective)\n_view(visualize(\n    AABB(Vec3f0(0), Vec3f0(vol_size)), :lines,\n    color =	lincolor,\n    model = offset,\n), camera = :perspective)\n\nparticle_vis[:color][1:n_particles] = map(1:n_particles) do i\n    xx = (i / n_particles) * 2pi\n    RGBA{Float32}((sin(xx) + 1) / 2, (cos(xx) + 1.0) / 2.0, 0.0, 0.1)\nend\n_view(visualize(\n    \"Smoke Particle\",\n    start_position = Point3f0(0), billboard = false,\n    model = translationmatrix(Vec3f0(0.1, -0.1, 4.1)) * rotationmatrix_x(Float32(0.5pi)),\n    relative_scale = 0.2f0, color = RGBA(1f0,1f0,1f0,1f0)\n), camera = :perspective)\n\n\n\n\noffset = translationmatrix(Vec3f0(4, 0, 0))\nmagnitude = map(x->sqrt(sum(x .* x)), isf.velocity);\nmagnitudecpu = Array(magnitude)\nmagnitudeviz = visualize(\n    magnitudecpu, :mip,\n    model = offset,\n    color_map = [RGBA(0f0, 0f0, 0f0, 0f0), RGBA(0.2f0, 0f0, 0.9f0, 0.9f0), RGBA(0.9f0, 0.2f0, 0f0, 0f0)],\n    dimensions = Vec3f0(vol_size)\n).children[]\n_view(magnitudeviz, camera = :perspective)\n_view(visualize(\n    AABB(Vec3f0(0), Vec3f0(vol_size)), :lines,\n    model = offset,\n    color =	lincolor,\n), camera = :perspective)\n_view(visualize(\n    \"Velocity Magnitude\",\n    start_position = Point3f0(0), billboard = false,\n    model = translationmatrix(Vec3f0(4.1, -0.1, -0.2)) * rotationmatrix_x(Float32(0.5pi)),\n    relative_scale = 0.2f0, color = RGBA(1f0,1f0,1f0,1f0)\n), camera = :perspective)\n\noffset = translationmatrix(Vec3f0(4, 0, 2))\ndirections = copy(isf.velocity)\ndirectionscpu = Array(directions)\ndirectionsviz = visualize(\n    reinterpret(Vec3f0, directionscpu),\n    model = offset,\n    color_map = [RGBA(0f0, 0f0, 0f0, 0f0), RGBA(0.2f0, 0f0, 0.9f0, 0.9f0), RGBA(0.9f0, 0.2f0, 0f0, 0f0)],\n    ranges = map(x-> 0:x, vol_size)\n).children[]\n_view(directionsviz, camera = :perspective)\n_view(visualize(\n    \"Velocity as Vectorfield\",\n    start_position = Point3f0(0), billboard = false,\n    model = translationmatrix(Vec3f0(4.1, -0.1, 4.1)) * rotationmatrix_x(Float32(0.5pi)),\n    relative_scale = 0.2f0, color = RGBA(1f0,1f0,1f0,1f0)\n), camera = :perspective)\n\n_view(visualize(\n    AABB(Vec3f0(0), Vec3f0(vol_size)), :lines,\n    model = offset,\n    color =	lincolor,\n), camera = :perspective)\n\ndt = isf.dt; d = isf.d\niter = 1\n\nio, buffer = GLVisualize.create_video_stream(\"test.mkv\", w)\n\n# main simulation loop\nwhile isopen(w)\n    t = iter * dt\n    # incompressible Schroedinger flow\n    schroedinger_flow!(isf, psi)\n    psi .= normalize_psi.(psi)\n    pressure_project!(isf, psi)\n\n    start = mod((iter - 1) * n_particles + 1, length(particles))\n    stop = start + n_particles - 1\n    add_particles!(particles, start:stop, nozzle_cen, nozzle_rad)\n    particle_vis[:color][start:stop] = map(1:n_particles) do i\n        xx = (i / n_particles) * 2pi\n        RGBA{Float32}((sin(xx) + 1) / 2, (cos(xx) + 1.0) / 2.0, mod(iter, 100) / 100, 0.1)\n    end\n    #constrain velocity\n    restrict_velocity!(isf, psi, kvec, isjetarr, omega*t)\n    pressure_project!(isf, psi)\n    velocity_one_form!(isf, psi, isf.hbar)\n    # inplace StaggeredSharp\n    dinv = inv.(isf.d)\n    broadcast!((x, y)-> x .* y, isf.velocity, isf.velocity, (dinv,))\n    staggered_advect!(particles, isf)\n\n    copy!(particlescpu, particles)\n    GLAbstraction.set_arg!(particle_vis, :position, reinterpret(Point3f0, particlescpu))\n\n    dircolor .= (x-> RGBA{Float32}(x[1] * 10f0, x[2] * 10f0, x[3] * 10f0, 1.0)).(isf.velocity)\n    copy!(dircolorcpu, dircolor)\n    GLAbstraction.set_arg!(dircolorviz, :volumedata, dircolorcpu)\n    magnitude .= (x->sqrt(sum(x .* x))).(isf.velocity)\n    copy!(magnitudecpu, magnitude)\n    GLAbstraction.set_arg!(magnitudeviz, :volumedata, magnitudecpu)\n\n    copy!(directionscpu, isf.velocity)\n    GLAbstraction.set_arg!(directionsviz, :rotation, vec(reinterpret(Vec3f0, directionscpu)))\n    GLWindow.poll_glfw()\n    GLWindow.reactive_run_till_now()\n    GLWindow.render_frame(w)\n    GLWindow.swapbuffers(w)\n    GLVisualize.add_frame!(io, w, buffer)\n    iter += 1\nend\nclose(io)\nGLWindow.destroy!(w)\npwd()\n<video width=\"100%\" controls>\n  <source src=\"smoke_simulation.webm\" type=\"video/webm\">\n  Your browser does not support the video tag. Please use a modern browser like Chrome or Firefox.\n</video>"
},

]}
