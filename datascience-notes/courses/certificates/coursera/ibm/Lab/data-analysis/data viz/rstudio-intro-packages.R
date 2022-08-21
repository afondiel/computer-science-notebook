# import datasets lib
library (datasets)

# load dataset
data(iris)

# view dataset content
View(iris) 

# display label classes
unique(iris$Species)


#installing new packages from command line

# install.packages("GGally", repos = "https://cran.r-project.org", type= "source") 
