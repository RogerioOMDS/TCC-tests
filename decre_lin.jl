using MultObjNLPModels, Random, Plots

include("funcaoAux.jl")

r(y, ŷ) = sum((y - ŷ).^2) / 2

n = 500
nb = 1

Random.seed!(0)
x = rand(n)
y = 2*x .+ 3 + randn(n) * 0.4
X = [ones(n) x]
nlp = LinearRegressionModel(X,y)

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
    y_pred = X * betas[:, j]
    res = r(y, y_pred)
    # println(res)
    append!(sol, res)
end

xg = 1:length(sol)

p = plot(size=(600,400), xlabel="Iterações", ylabel="Resíduo", title="$(n) pontos", palette=:jet)
# xlims!(p, 0, 100)
# ylims!(p, 0, 20)
plot!(p, xg, sol, leg=false)

png(p, "decre_linear$(n)_batch$(nb)")