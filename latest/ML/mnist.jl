using Flux, MNIST
using Flux: onehotbatch, argmax, mse, throttle
using Base.Iterators: repeated

x, y = traindata()
y = onehotbatch(y, 0:9)

m = Chain(
    Dense(28^2, 32, σ),
    Dense(32, 10),
    softmax
)

using CLArrays
using GPUArrays: JLArray
using CLArrays: GlobalArray, GlobalPointer, PreDeviceArray
using Flux: OneHotMatrix

cl(x) = x
cl(x::JLArray) = x
cl(xs::AbstractArray) = isbits(xs) ? xs : JLArray(xs)
cl(xs::Flux.TrackedArray) = Flux.TrackedArray(xs.f, cl(xs.data), Base.RefValue(cl(Flux.Tracker.grad(xs))))
cl(xs::Flux.OneHotMatrix) = Flux.OneHotMatrix(cl(xs.data))

CLArrays.kernel_convert(x::OneHotMatrix{T}) where T <: CLArray = OneHotMatrix(CLArrays.kernel_convert(x.data))
CLArrays.predevice_type(::Type{OneHotMatrix{T}}) where T <: GlobalArray = OneHotMatrix{CLArrays.predevice_type(T)}
CLArrays.device_type(x::OneHotMatrix{T}) where T <: CLArray = OneHotMatrix{CLArrays.device_type(x.data)}
CLArrays.reconstruct(x::OneHotMatrix{T}, ptr::GlobalPointer) where T <: PreDeviceArray = OneHotMatrix(CLArrays.reconstruct(x.data, ptr))
GPUArrays.arg_length(x::OneHotMatrix{T}) where T <: GPUArrays.GPUArray = UInt32.(size(x))
GPUArrays.to_device(state, x::OneHotMatrix{<: JLArray}) = OneHotMatrix(GPUArrays.to_device(state, x.data))
x, y = cl(x), cl(y)
m = mapparams(cl, m)
loss(x, y) = mse(m(x), y)

dataset = repeated((x, y), 500)
evalcb = () -> @show(loss(x, y))
opt = SGD(params(m), 1)

Flux.train!(loss, dataset, opt, cb = throttle(evalcb, 10))

# Check the prediction for the first digit
argmax(m(x[:,1]), 0:9) == argmax(y[:,1], 0:9)
