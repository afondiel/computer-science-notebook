library(datasets)
data(iris)
View(iris)

unique(iris$Species)


library(GGally)

ggpairs(iris, mapping=ggplot2::aes(colour = Species))