using MultObjNLPModels, Random, Plots, LinearAlgebra, NLPModels
Random.seed!(0)

h(β, x) = 1 / (1+exp(-β[1] - β[2] * x))

n = 10 # Number data

# LOGISTIC REGRESSION

x = rand(n,2)
y = [x[i,1]^2 + x[i,2]^2 > 0.7 + randn() * 0.2 ? 1.0 : 0.0 for i=1:n]
X = [ones(n) x[:,1]]

L(β) = -sum(y[i] * log(h(β, x[i])) + (1 - y[i])*log(1 - h(β,x[i])) for i=1:n) / n 

q = contour(
    range(-5, 5, length=50),
    range(-1, 7, length=50),
    (x,y) -> L([x,y]),
    levels=100, leg=false
)

nlp = LogisticRegressionModel(X, y)
iter, β, betas, gamas, q = sthocastic_gradient(q, nlp, batch_size=1, γ0 = 10.0, max_iter = 100)


# LINEAR REGRESSION

# x = rand(n)
# y = 2*x .+ 3 + randn(n) * 0.4
# X = [ones(n) x]
# G(β) = sum( (β[1] + β[2] * x[i] - y[i])^2 for i=1:n) / 2

# q = contour(
#     range(-2, 5, length=50),
#     range(-5, 5, length=50),
#     (x,y) -> G([x,y]),
#     levels=100, leg=false, title="Gradiente Descendente"
# )

# nlp = LinearRegressionModel(X, y)

# iter, β, betas, gamas, q = sthocastic_gradient(q, nlp, batch_size=10, γ0 = 3.0, max_iter = 7)

png(q, "image") 
