using CSV, DataFrames, MultObjNLPModels, BenchmarkTools
include("funcaoAux.jl")

df = CSV.File("new_biddings.csv", type=Float32)
df = DataFrame(df)

X_train = df[1:300000,1:88]
Xtr = zeros(300000,88)
Xtr .= X_train

y_train = df[1:300000,89]


X_test = df[300001:499999, 1:88]
Xts = zeros(199999, 88)
Xts .= X_test

y_test = df[300001:499999, 89]

nlp = LogisticRegressionModel(Xtr, y_train)

# b = @benchmark sthocastic_gradient(nlp)
# run(b, eval = 1)


output = sthocastic_gradient(nlp, learning_rate=:exponential)
β = output.solution
# iter, β, betas, gamas = output

y_pred = σ.(Xts*β)

y_class = classificador(y_pred)

accuracy(y_class, y_test)