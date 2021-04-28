using CSV
using TextAnalysis
using Statistics
using MultivariateStats
using Distributions
using NaiveBayes
import Clustering
import StatsBase

#SMSSpamCollection.csv


dt = CSV.read("SMSSpamCollection.csv";delim="\t")

arr = [] 

#fill array with dictionaries with the classification and the message 
 for i in 1:5571
    #println(dt[1][i]* " " * dt[2][i])
    push!(arr, Dict("classification" => dt[1][i], "message" => StringDocument(dt[2][i])))
end

mess = [x["message"] for x in arr] #get only the messages 

c = Corpus(mess)

remove_case!(c)
prepare!(c, strip_punctuation)
stem!(c)
update_lexicon!(c)

mtx = collect(transpose(dtm(c)))
"""
make a subset but choosing the first 5000 columns for training 
and the last 571 for testing, assume that the data is randomized
"""

train_mtx = mtx[:,1:5000]
test_mtx = mtx[:,5001:5571]

y = [dt[1][i] for i in 1:5000] #need a vector of strings with the spam or ham

m = MultinomialNB(unique(y), 7210)
fit(m, train_mtx, y) #fit the model using naive bayes

"""
predict the text subset
"""
pred = predict(m, test_mtx) #predicted value for 
tru_val = [dt[1][i] for i in 5001:5571] #what they should actually be 

accuracy = sum(pred .== tru_val) / 571 

println("the accuracy is " * string(accuracy)) 



"""
docTrmMtx = tf(dtm(c)) #need to use this so its a frequency took a long time and had to read documenation to figure out 

keepIndex = [] #will be used to store indices of the words that appear more than once 

for i in 1:size(docTrmMtx)[2]
    col = docTrmMtx[1:end, i]
    numZero = size(col.nzind)[1]
    if(numZero > 2)
        push!(keepIndex, i)
    end
end


new_dtm = docTrmMtx[:,keepIndex] #a document terms matrix with just the terms that appear more than twice

PCAreducedMatrix = collect(transpose(new_dtm))

pca1 = fit(PCA, PCAreducedMatrix, maxoutdim = 300) #played around and chose 150 dimensions 



nclusters = 2
clusterMatrix = transform(pca1, PCAreducedMatrix)

k2 = Clustering.kmeans(clusterMatrix, nclusters)

println(StatsBase.counts(k2.assignments))


function acc()
correct = 0 

for i in 1:5571
    if(k2.assignments[i] == 2 && arr[i]["classification"] == "ham" )
        correct += 1
    end
end

println(correct)

end 

acc()
"""
