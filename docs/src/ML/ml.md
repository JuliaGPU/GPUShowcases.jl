# MNIST

[From the model zoo](https://github.com/FluxML/model-zoo/blob/master/mnist/mnist.jl)

```Julia
using Flux, MLDatasets, CuArrays
using Flux: onehotbatch, argmax, crossentropy, throttle, onecold
using Base.Iterators: repeated

x, y = x, y = MNIST.traindata()
x = reshape(x,28*28,:) |> gpu
y = onehotbatch(y, 0:9) |> gpu

m = Chain(
    Dense(28^2, 32, σ),
    Dense(32, 10),
    softmax
) |> gpu


loss(x, y) = crossentropy(m(x), y)

dataset = repeated((x, y), 500)
evalcb = () -> @show(loss(x, y))
opt = Descent()

Flux.train!(loss, params(m), dataset, opt, cb = throttle(evalcb, 10))

# Check the prediction for the first digit
onecold(m(x[:,1]), 0:9) == onecold(y[:,1], 0:9)
```
