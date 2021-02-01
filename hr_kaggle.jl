using CSV, DataFrames, MultObjNLPModels, BenchmarkTools
include("funcaoAux.jl")

df = CSV.File("HR_data.csv")
df = DataFrame(df)
df

X_train = df[1:11000,2:end]
Xtr = zeros(11000,9)
Xtr .= X_train

X_test = df[11001:end, 2:end]
Xts = zeros(3999, 9)
Xts .= X_test

y_train = df[1:11000,1]

y_test = df[11001:end, 1]


nlp = LogisticRegressionModel(Xtr, y_train)

output = sthocastic_gradient(nlp, max_iter = 100, γ0=3e-2)

println(output)

β = output.solution

y_pred = σ.(Xts*β)

y_class = classificador(y_pred)

@info("",sum(y_test))
@info("",sum(y_pred))
@info("",sum(y_class))

# println(y_pred)
# println("y_class = $(y_class)")
# println("y_test = $(y_test)")
@info("",accuracy(y_class, y_test))

sub = y_class - y_test

fp_num = [x > 0 ? 1 : 0 for x in sub]
fp = sum(fp_num)

fn_num = [x ≥ 0 ? 0 : 1 for x in sub]
fn = sum(fn_num)

sum_sub = sub + y_class
tp_num = [x == 1 ? 1 : 0 for x in sum_sub]
tp = sum(tp_num)

tn = length(y_test) - tp - fn - fp

precision_score = tp / (tp + fp)

recall_score = tp / (tp + fn)

accuracy_score = (tp + tn) / (fp + fn + tp + tn)

f1_score = (2 * precision_score * recall_score ) / (precision_score + recall_score)

@info("", fp, fn, tp, tn, precision_score, recall_score, accuracy_score, f1_score)