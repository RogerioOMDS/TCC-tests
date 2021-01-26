using MultObjNLPModels, Random, Plots

include("funcaoAux.jl")

r(y, ŷ) = - sum(y[i] * log(ŷ[i]) + (1 - y[i]) * log(1 - ŷ[i]) for i=1:length(y)) / length(y)
σ(z) = 1 / (1 + exp(-z))

n = 100
nb = 100

Random.seed!(0)
x = rand(n,2)
y = [x[i,1]^2 + x[i,2]^2 > 0.7 + randn() * 0.2 ? 1.0 : 0.0 for i=1:n]
X = [ones(n) x]
nlp = LogisticRegressionModel(X,y)

output = sthocastic_gradient(nlp, batch_size=nb)
# β = output.solution

β = output[2]
betas = output[3]
gamas = output[4]

b =  div(length(betas), size(β,1))
betas = reshape(betas, size(β,1), b)

##################

sol = []

for j=1:b
    y_pred = σ.(X * betas[:, j])
    res = r(y, y_pred)
    append!(sol, res)
end

xg = 1:length(sol)

p = plot(size=(600,400), xlabel="Iterações", ylabel="Resíduo", title="$(n) pontos", palette=:jet)
# xlims!(p, 0, 100)
# ylims!(p, 0, 20)
plot!(p, xg, sol, leg=false)

png(p, "decre_logi$(n)_batch$(nb)")